#! /usr/bin/env python
# encoding: utf-8
import logging
import sys
from PIL import Image
try:
    from .config_manager import Config, default_config
    from .config_util import load_config
    from .line_breaker import LineCluster, LineCut
    from .page_elements2 import PageElements as PE2, PageElementsOptions as PE2Options
    from .scale_invariant_preprocessing import apply_scale_invariant_preprocessing, apply_band_preprocessing
    from .recognize import cls, rbfcls
    # Restore original sophisticated recognition pipeline
    from .recognize import recognize_chars_probout, recognize_chars_hmm
    from .recognize import hmm_recognize_bigram
    from .segment import Segmenter, combine_many_boxes
except ImportError:
    from config_manager import Config, default_config
    from config_util import load_config
    from line_breaker import LineCluster, LineCut
    from page_elements2 import PageElements as PE2, PageElementsOptions as PE2Options
    from scale_invariant_preprocessing import apply_scale_invariant_preprocessing, apply_band_preprocessing
    from recognize import cls, rbfcls
    # Restore original sophisticated recognition pipeline
    from recognize import recognize_chars_probout, recognize_chars_hmm
    from recognize import hmm_recognize_bigram
    from segment import Segmenter, combine_many_boxes
import numpy as np
import argparse
try:
    from .utils_extra.scantailor_multicore import run_scantailor
    from .fast_utils import fadd_padding
except (ImportError, ModuleNotFoundError):
    try:
        from utils_extra.scantailor_multicore import run_scantailor
        from fast_utils import fadd_padding
    except (ImportError, ModuleNotFoundError):
        run_scantailor = None
        fadd_padding = None
import codecs
import os
try:
    from .yik import word_parts_set
    from .root_based_finder import is_non_std
except ImportError:
    from yik import word_parts_set
    from root_based_finder import is_non_std
try:
    from .termset import syllables
except ImportError:
    from termset import syllables
import tempfile

import platform

class FailedPageException(Exception):
    pass

class PageRecognizer(object):
    def __init__(self, imagefile, conf, page_info={}, retries=0, text=False):
        self.confpath = conf.path   # used by the line_cut→line_cluster retry
        self.conf = conf.conf
        self.imagefile = imagefile
        self.text = text
        raw_page_array = np.asarray(Image.open(imagefile).convert('L'))/255

        # Diagnostic: log image info for debugging first-entry issues
        print(f"[NAMSEL-DIAG] Processing: {os.path.basename(imagefile)}, shape: {raw_page_array.shape}, "
              f"min: {raw_page_array.min():.3f}, max: {raw_page_array.max():.3f}, "
              f"text_pixels: {np.sum(raw_page_array < 0.5)}")

        # Apply scale-invariant preprocessing for optimal OCR performance
        if self.conf.get('debug_output', False):
            print(f"[PREPROCESSING] Applying scale-invariant preprocessing to {imagefile}")
        if self.conf.get('force_single_line', False):
            self.page_array = apply_band_preprocessing(raw_page_array)
        else:
            self.page_array = apply_scale_invariant_preprocessing(raw_page_array)
        # Check if image is truly blank (no dark pixels at all)
        # Use threshold < 0.5 to find text pixels instead of .all() which fails
        # when min pixel value is slightly above 0 (e.g. 0.086)
        text_pixel_count = np.sum(self.page_array < 0.5)
        if text_pixel_count == 0:
            print("[NAMSEL-DIAG] WARNING: No text pixels found — image appears blank after preprocessing!")
            self.conf['line_break_method'] = 'line_cut'

        # Determine whether a page is of type book or pecha, and the line-break method.
        self.line_break_method = self.conf.get('line_break_method', None)
        self.page_type = self.conf.get('page_type', None)
        self.retries = retries
        self.page_info = page_info

        self.imgheight = self.page_array.shape[0]
        self.imgwidth = self.page_array.shape[1]

        self._resolve_line_break_and_page_type()

        self.conf['page_type'] = self.page_type
        self.conf['line_break_method'] = self.line_break_method
        if self.line_break_method == 'line_cluster' and self.page_type != 'pecha':
            self.page_type = 'pecha'   # line_cluster requires pecha
        self.detect_o = self.conf.get('detect_o', False)

    def _resolve_line_break_and_page_type(self):
        """Fill in line_break_method / page_type when unset: auto-detect from the
        image aspect ratio (wide → pecha/line_cluster, else book/line_cut), or derive
        line_break_method from a given page_type."""
        if not self.line_break_method and not self.page_type:
            if self.page_array.shape[1] > 2 * self.page_array.shape[0]:
                self.line_break_method, self.page_type = 'line_cluster', 'pecha'
            else:
                self.line_break_method, self.page_type = 'line_cut', 'book'
        elif not self.line_break_method:
            self.line_break_method = 'line_cluster' if self.page_type == 'pecha' else 'line_cut'

    ################################
    # The main recognition pipeline
    ################################
    def get_page_elements(self):
        '''PageElements (PE2) does a first-pass segmentation of blob (characters/punc)
        ona page, gathers information about width of page objects,
        isolates body text of pecha-style pages, and determines the
        number of lines on a page for use in line breaking'''
        self.shapes = PE2(self.page_array, cls, PE2Options(
                     page_type=self.page_type,
                     low_ink=self.conf['low_ink'],
                     flpath=self.page_info.get('flname',''),
                     detect_o=self.detect_o,
                     clear_hr=self.conf.get('clear_hr', False),
                     force_single_line=self.conf.get('force_single_line', False)))
        self.shapes.conf = self.conf
        if self.page_type == 'pecha' or self.line_break_method == 'line_cluster':
            if not hasattr(self.shapes, 'num_lines'):
                print('Error. This page can not be processed. Please inspect the image for problems')
                raise FailedPageException('The page ({}) you are attempting to process failed'.format(self.imagefile))
            # num_lines can be 0 for a single-line / valley-less page; KMeans in
            # LineCluster requires n_clusters >= 1, so floor k_groups at 1.
            self.k_groups = max(1, self.shapes.num_lines)
            self.shapes.viterbi_post = self.conf['viterbi_postprocessing']

    def extract_lines(self):
        '''Identify lines on a page of text'''
        if self.line_break_method == 'line_cut':
            self.line_info = LineCut(self.shapes)
            if not self.line_info:  # immediately skip to re-run with LineCluster
                sys.exit()
        elif self.line_break_method == 'line_cluster':
            self.line_info = LineCluster(self.shapes, k=self.k_groups)

        if hasattr(self, 'line_info') and self.line_info:
            self.line_info.rbfcls = rbfcls

    def generate_segmentation(self):
        if hasattr(self, 'line_info') and self.line_info:
            self.segmentation = Segmenter(self.line_info)
        else:
            if default_config.get('debug_output', False):
                print("[DEBUG] No line_info available, skipping segmentation")
            self.segmentation = None

    def recognize_page(self, text=False):
        try:
            self.get_page_elements()
            self.extract_lines()
        except:
            import traceback;traceback.print_exc()
            self.results = []
            return self.results

        self.generate_segmentation()

        if not self.segmentation:
            if default_config.get('debug_output', False):
                print("[DEBUG] No segmentation available, returning empty results")
            self.results = []
            return self.results

        conf = self.conf
        results = []
        out = None
        try:
            if conf['viterbi_postprocessing']:
                # Should only be called from within a non-viterbi run.
                prob, results = hmm_recognize_bigram(self.segmentation)
                return prob, results

            results = self._run_recognizer(conf)
            out, results = self._build_output(results, text)
            if platform.system() != "Windows" and self.conf.get('debug_output', False):
                print(out)
            self.results = results
            return results
        except:
            import traceback; traceback.print_exc()
            return self._handle_recognition_failure(conf, results, out, text)

    def _retry_with_line_cluster(self, text):
        """One retry with line_cluster/pecha when line_cut produced no result."""
        try:
            pr = PageRecognizer(self.imagefile, Config(path=self.confpath, line_break_method='line_cluster', page_type='pecha'), page_info=self.page_info, retries=1, text=text)
            return pr.recognize_page(text=text)
        except:
            logging.info('Exited after failure of second run.')
            return []

    def _run_recognizer(self, conf):
        """Dispatch to the configured recognizer, then optional viterbi postprocess."""
        results = []
        if conf['recognizer'] == 'probout':
            results = recognize_chars_probout(self.segmentation)
        elif conf['recognizer'] == 'hmm':
            results = recognize_chars_hmm(self.segmentation)
        if conf['postprocess']:
            results = self.viterbi_post_process(self.page_array, results)
        return results

    def _build_output(self, results, text):
        """Flatten the per-line char results into a string (text) or utf-8 bytes.
        Returns (out, results)."""
        output = []
        for n, line in enumerate(results):
            for m, k in enumerate(line):
                if isinstance(k[-1], int):
                    print((n, m, k))
                    self.page_array[k[1]:k[1]+k[3], k[0]:k[0]+k[2]] = 0
                    Image.fromarray(self.page_array * 255).show()
                output.append(k[-1])
            output.append('\n')
        if text:
            out = ''.join(output)
            return out, out
        return ''.join(output).encode('utf-8'), results

    def _handle_recognition_failure(self, conf, results, out, text):
        """Failure path: retry line_cut→line_cluster once, else log + return what we have."""
        if not results and not conf['viterbi_postprocessing']:
            filename = self.page_info.get('flname', 'unknown_file')
            print(('WARNING', '*'*40))
            print((filename, 'failed to return a result.'))
            print(('WARNING', '*'*40))
            print()
            if self.line_break_method == 'line_cut' and self.retries < 1:
                return self._retry_with_line_cluster(text)
        if not conf['viterbi_postprocessing']:
            if not results:
                logging.info('***** No OCR output for %s *****' % self.page_info['flname'])
            if text:
                results = out
            self.results = results
            return results


    #############################
    # Helper and debug methods
    #############################

    def generate_line_imgs(self):
        pass

    #############################
    ## Experimental
    #############################
    def _vpp_run_hmm(self, arr):
        """Re-run the line-cut HMM recognizer on a padded syllable sub-image;
        returns (prob, hmm_res), or (0, '') when the run errors out."""
        try:
            temp_dir = tempfile.mkdtemp()
            tmpimg = os.path.join(temp_dir, 'tmp.tif')
            Image.fromarray(arr*255).convert('L').save(tmpimg)
            pgrec = PageRecognizer(tmpimg, Config(line_break_method='line_cut', page_type='book', postprocess=False, viterbi_postprocessing=True, clear_hr=False, detect_o=False))
            prob, hmm_res = pgrec.recognize_page()
            os.remove(tmpimg)
            os.removedirs(temp_dir)
            return prob, hmm_res
        except TypeError:
            if self.conf.get('debug_output', False):
                print('HMM run exited with an error.')
            return 0, ''

    def _vpp_fix_syllable(self, img_arr, syllable):
        """Try to correct one non-standard syllable via the HMM. Returns the
        corrected box [x,y,w,h,prob,hmm_res], or None to keep the syllable as-is
        (either it was already standard, or the HMM produced nothing usable)."""
        syl_str = ''.join(s[-1] for s in syllable)
        if not (is_non_std(syl_str) and syl_str not in syllables):
            return None
        if self.conf.get('debug_output', False):
            print((syl_str, 'HAS PROBLEMS. TRYING TO FIX'))
        bx = list(combine_many_boxes([ch[0:4] for ch in syllable]))
        arr = fadd_padding(img_arr[bx[1]:bx[1]+bx[3], bx[0]:bx[0]+bx[2]], 3)
        prob, hmm_res = self._vpp_run_hmm(arr)
        logging.info('VPP Correction: %s\t%s' % (syl_str, hmm_res))
        if prob == 0 and hmm_res == '':
            if self.conf.get('debug_output', False):
                print('hit problem. using unmodified output')
            return None
        bx.append(prob)
        bx.append(hmm_res)
        return bx

    def viterbi_post_process(self, img_arr, results):
        '''Go through all results and attempts to correct invalid syllables'''
        final = [[] for i in range(len(results))]
        for i, line in enumerate(results):
            syllable = []
            for j, char in enumerate(line):
                if char[-1] in '་། ' or not word_parts_set.intersection(char[-1]) or j == len(line)-1:
                    if syllable:
                        fixed = self._vpp_fix_syllable(img_arr, syllable)
                        if fixed is None:
                            final[i].extend(syllable)
                        else:
                            final[i].append(fixed)
                    final[i].append(char)
                    syllable = []
                else:
                    syllable.append(char)
            if syllable:
                final[i].extend(syllable)

        return final

def generate_formatted_page(page_info):
    pass

def run_recognize(imagepath):
    global args
    command_args = args
    if command_args.conf:
        conf_dict = load_config(command_args.conf)
    else:
        conf_dict = default_config

    # Override any confs with command line versions
    for key in conf_dict:

        if not hasattr(command_args, key):
            continue
        val = getattr(command_args, key)
        # For boolean flags, set them regardless of truthiness
        # For other values, only set if truthy
        if key in ('debug_output', 'force_single_line') or val:
            conf_dict[key] = val

    rec = PageRecognizer(imagepath, conf=Config(**conf_dict))
    if args.format == 'text':
        text = True
    else:
        text = False
    return rec.recognize_page(text=text)

def run_recognize_remote(imagepath, conf_dict, text=False):
    rec = PageRecognizer(imagepath, conf=Config(**conf_dict))
    results = rec.recognize_page(text=text)
    return results

def _build_arg_parser():
    """Construct the Namsel CLI argument parser (action + image path + config
    overrides + scantailor preprocessing options)."""
    parser = argparse.ArgumentParser(description='Namsel OCR')

    action_choices = ['preprocess', 'recognize-page', 'isolate-lines', 'view-page-info',
                      'recognize-volume']
    parser.add_argument('action', type=str, choices=action_choices,
                        help='The Namsel function to be executed')
    parser.add_argument('imagepath', type=str, help="Path to jpeg, tiff, or png image (or a folder containing them, in the case of recognize-volume)")
    parser.add_argument('--conf', type=str, help='Path to a valid configuration file')
    parser.add_argument('--format', type=str, choices=['text', 'page-info'], help='Format returned by the recogizer')
    parser.add_argument('--outfile', type=str, help='Name of the file saved in the ocr_ouput folder. If not specified, filename will be "ocr_output.txt"')
    # Config override options
    confgroup = parser.add_argument_group('Config', 'Namsel options')
    confgroup.add_argument('--page_type', type=str, choices=['pecha', 'book'], help='Type of page')
    confgroup.add_argument('--line_break_method', type=str, choices=['line_cluster', 'line_cut'],
                           help='Line breaking method. Use line_cluster for page type "pecha"')
    confgroup.add_argument('--recognizer', type=str, choices=['hmm', 'probout'],
                           help='The recognizer to use. Use HMM unless page contains many hard-to-segment and unusual characters')
    confgroup.add_argument('--break_width', type=float, help='Threshold value to determine segmentation, measured in stdev above the mean char width')
    confgroup.add_argument('--segmenter', type=str, help='Type of segmenter to use', choices=['stochastic', 'experimental'])
    confgroup.add_argument('--low_ink', type=bool, help='Attempt to enhance results for poorly inked prints')
    confgroup.add_argument('--line_cluster_pos', type=str, choices=['top', 'center'])
    confgroup.add_argument('--postprocess', type=bool, help='Run viterbi post-processing')
    confgroup.add_argument('--detect_o', type=bool, help='Detect and set aside na-ro vowels in first pass recognition')
    confgroup.add_argument('--clear_hr', type=bool, help='Clear all content above a horizontal rule on top of a page')
    confgroup.add_argument('--line_cut_inflation', type=int, help='The number of iterations to use when dilating image in line breaking. Increase this value when you want to blob things together')
    confgroup.add_argument('--debug_output', action='store_true', help='Enable debug output (warnings, preprocessing messages, etc.)')
    confgroup.add_argument('--enable_interactive_segmentation', action='store_true', help='Enable interactive manual segmentation for wide characters')
    confgroup.add_argument('--force_single_line', action='store_true', help='Treat input as a single pre-segmented line')

    scantailor_conf = parser.add_argument_group('Scantailor', 'Preprocessing options')
    scantailor_conf.add_argument('--layout', choices=['single', 'double'], type=str,
                                 help='Option for telling scantailor to expect double or single pages')

    scantailor_conf.add_argument('--threshold', type=int, help="The amount of thinning or thickening of the output of scantailor. Good values are -40 to 40 (for thinning and thickening respectively)")

    return parser


def _action_recognize_page(args, outfilename):
    """Run single-page recognition and write the text output file."""
    results = run_recognize(args.imagepath)
    if args.format == 'text':
        with codecs.open(outfilename, 'w', 'utf-8') as outfile:
            outmessage = '''OCR text\n\n'''
            outfile.write(outmessage)
            outfile.write(os.path.basename(args.imagepath)+'\n')
            if not isinstance(results, str):
                results = 'No content captured for this image'
                print('****************')
                print(results)
                print("Saving empty page to output")
                print('****************')
            outfile.write(results)



def _action_recognize_volume(args, outfilename):
    """Recognize every tif in a folder (multiprocessing off Windows) and write
    the combined text output file."""
    import sys
    if platform.system() != "Windows":
        import multiprocessing

    import glob
    if not os.path.isdir(args.imagepath):
        print('Error: You must specify the name of a directory containing tif images in order to recognize a volume')
        sys.exit()

    if platform.system() != "Windows":
        pool = multiprocessing.Pool()

    pages = glob.glob(os.path.join(args.imagepath, '*tif'))
    pages.sort()

    if platform.system() == "Windows":
        results = list(map(run_recognize,  pages))
    else:
        results = pool.map(run_recognize,  pages)

    if args.format == 'text':

        with codecs.open(outfilename, 'w', 'utf-8') as outfile:
            outmessage = '''OCR text\n\n'''
            outfile.write(outmessage)

            for k, r in enumerate(results):
                outfile.write(os.path.basename(pages[k])+'\n')
                if not isinstance(results, str):
                    if isinstance(results, bytes):
                        results = results.decode('utf-8')
                    else:
                        results = 'No content captured for this image'

                print(">>> OCR Result:", results)
                outfile.write(r.decode('utf-8') + '\n\n')


def main(argv=None):
    """Main function that can be called programmatically"""
    import sys
    if argv is None:
        argv = sys.argv[1:]
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    if not os.path.exists('ocr_results'):
        os.mkdir('ocr_results')
    outfilename = args.outfile if args.outfile else 'ocr_output.txt'
    if args.action == 'recognize-page':
        _action_recognize_page(args, outfilename)
    elif args.action == 'recognize-volume':
        _action_recognize_volume(args, outfilename)
    elif args.action == 'preprocess':
        run_scantailor(args.imagepath, args.threshold, layout=args.layout)


if __name__ == '__main__':
    main()
