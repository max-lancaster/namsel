#! /usr/bin/python
# encoding: utf-8
'''
    Determine how coherent a section of text is using scores that rate
    whether syllables and terms are in a dictionary, whether they follow
    standard writing rules, etc...

    TODO: the individual scoring functions could stand to be refactored since
    they employ almost identical patterns.

'''
    

from .root_based_finder import is_non_std, word_parts
from .termset import syllables as termset
import re
import codecs
word_parts = ''.join(list(word_parts))

reg = re.compile(r'[^%s]*' % word_parts)

LOG = False

class TextScores():
    def __init__(self, text, multiline=False):
        self.text = text
        self.multiline = multiline
        
        # Call main func to set all scores
        self.set_text_scores(text, multiline)
    
    def __repr__(self): 
        return '%s' % str('\n'.join('%s : %s' % (k, repr(v)) for (k, v) in self.__dict__.items()))
    
    @classmethod
    def _dict_score(self, syllables, multiline=False):
        '''Look up syllables in dict. Return percentage that were found. 
        Expects syllables argument to be a list'''
        
        results = {}
        self.not_in_dict = set()
        if multiline:
            syl_len = float(sum(len(i) for i in syllables)) - len(syllables) + 1
            total_found = 0.
            scores = []
            for i, line in enumerate(syllables):
                # account for extra empty strs that are an artifact of splitting
                if i == 0:
                    c = 0
                else:
                    c = 1.
    
                line_len = float(len(line)) - c
                found = 0.
                for syl in line:
                    if syl and syl in termset:
                        found += 1.
                    elif syl:
                        self.not_in_dict.add(syl)
                scores.append(found/(line_len or 1)) #annoying workaround to avoid dividing by zero
                total_found += found
            results['lines_dict_scores'] = scores
            results['all_text_dict_score'] = total_found / syl_len
        else:
            found = 0.
            for i in syllables:
                if i and i in termset:
                    found += 1.
                elif i:
                    self.not_in_dict.add(syl)
            results['all_text_dict_score'] = found/float(len(syllables))
                
            # Record letter groups that could possibly be added to the termset
            if LOG:
                if i not in termset and not is_non_std(i):
                    dict_candidates = codecs.open('dictionary_candidates.txt', 'a', 'utf-8')
                    dict_candidates.write(i+'\n')
                    dict_candidates.flush()
        return results
    
    @classmethod
    def _non_std_score(self, syllables, multiline=False):
        '''Check if a word is non-std (i.e. doesn't follow rules that dictate 
        how Tibetan letters can be combined)'''
        
        results = {}
        self.nonstd = set()
        if multiline:
            # In this case, syllables is list of lists containing syllables
            # the len(i)-1 and +1 attempt to account for the '' blank str that 
            # splitting into lines leaves.
            syl_len = float(sum(len(i) for i in syllables)) - len(syllables) + 1
            total_std = 0.
            scores = []
            for i, line in enumerate(syllables):
                if i == 0:
                    c = 0
                else:
                    c = 1.
                line_len = float(len(line)) - 1.
                std = 0.
                for syl in line:
                    if syl and not is_non_std(syl):
                        std += 1.
                    elif syl: self.nonstd.add(syl)
                
                scores.append(std/(line_len or 1))
                total_std += std
            results['lines_nonstd_scores'] = scores
            results['all_text_nonstd_score'] = total_std / syl_len
            
        else:
            std = 0.
            for i in syllables:
                if not is_non_std(i):
                    std += 1.
                else:
                    self.nonstd.add(i)
            results['all_text_nss'] = std/float(len(syllables))
    #        else:
    #            print i
        return results

    def _target_score(self, syllables, multiline=False):
        '''
        This is a hybrid of non-std and dictionary scores. A syllable is
        checked to see if it is both non-standard and not in the dictionary.
        If it is, it is presumed to be mispelled/incorrect
        '''
        
        results = {}
        self.invalid = []
        self.not_in_dict = []
        self.nonstd = set()
        if multiline:
            # In this case, syllables is list of lists containing syllables
            # the len(i)-1 and +1 attempt to account for the '' blank str that 
            # splitting into lines leaves.
            syl_len = float(sum(len(i) for i in syllables)) - len(syllables) + 1
            total_nstd = 0.
            scores = []
            for i, line in enumerate(syllables):
                if i == 0:
                    c = 0
                else:
                    c = 1.
                line_len = float(len(line)) - 1.
                nstd = 0.
                for syl in line:
                    is_ns = is_non_std(syl)
                    # outside all the circles of the venn diagram.
                    # use this for scoring
                    if syl and is_ns and not syl in termset:
                        nstd += 1.
                        self.invalid.append(syl)
                    
                    # one circle of the venn diagram. may overlap w/ other circle
                    # Since this score is default, want to make sure this
                    # info is captured somewhere
                    if syl and not is_ns and syl not in termset:
                        self.not_in_dict.append(syl)

                    # The other circle
                    if is_ns: self.nonstd.add(syl)
                
                scr = 1. - (nstd/(line_len or 1))
                if scr < 0: scr=0.
                scores.append(scr)
                total_nstd += nstd
            results['lines_target_scores'] = scores
            results['page_target_score'] = 1. - (total_nstd / syl_len)
            results['non_standard_count'] = nstd
        else:
            nstd = 0.
            for i in syllables:
                if i and is_non_std(i) and i not in termset:
                    nstd += 1.
                    self.invalid.append(i)

            results['target_score'] = 1.-(nstd/float(len(syllables)))
            results['non_standard_count'] = nstd # Number of non standard syllables
    #        else:
    #            print i
        return results
    
    def _punc_syllable_ratio(self, text, multiline=True):
        '''Only provide line by line results if multiline is True (not results
        for the entire page)'''
        results = {}
        if multiline:
            lines = text.split('\n')
            scores = []
            for i,l in enumerate(lines):
                total = 0.
                for j in '་།':
                    total += l.count(j)
#                scores.append(total/(float(len(l)-1) or 1)) # no divide by zero
                scores.append(total/(float(len(l)))) # no divide by zero
            results['lines_punc_syl_ratios'] = scores
        else:
            len_syl = len(reg.split(text))
            total = 0.
            for j in '་།':
                total += text.count(j)
            results['all_text_punc_syl_ratio'] = total / len_syl
            
        return results
    
    def set_text_scores(self, res_str, multiline=False):
        '''
        Check a string of text for coherency based on termset and spelling rules
        
        multiline is useful for things like pecha pages where you want to score
        on a more granular basis
        '''
        results = {}
        if multiline:
            text = res_str.split('\n')
            from string import whitespace
            whitespace = set(whitespace) 
            syllables = [reg.split(t) for t in text if not whitespace.issuperset(t)]
        else:
            syllables = reg.split(res_str)
            
        results.update(self._target_score(syllables, multiline=multiline)) 
        self.__dict__.update(results)
    
if __name__ == '__main__':
    sample = '''གང་དྲན་ཆོས་སྐུར་ཤེས། །སྨིག་རྒྱུ་ནམ་མཁར་ཤེས་པ་འདྲ། །བརྟུལ་ཞགས་མཆོག་གི་སྤྱོད་པ་ནི། །མཚམས་མེད་ལྔ་དང་མི་དགེ་བཅུ། །ཆགས་སྡང་ལྔ་ལ་ལོངས་སྤྱད་ཀྱང་། །ལྟ་བ་རྟོགས་པའི་གྲངས་སུ་འགྱུར། །
ལུང་རིགས་མི་འགལ་རྒྱུད་གནས་དང་། །འབོལ་ཞིང་ཞི་ལ་འཇམ་འདོད་པས། །ངེས་པའི་གདམས་ངག་མ་གོ་ན། །མཁས་ཀྱང་རྨོངས་པ་སྙིང་རེ་རྗེ། །དེ་ཡང་འདུས་པའི་སྙིང་པོ་ཧོན། །བགེཀོསས་མེད་ཡེ་ཤེས་རང་
ཤར་བ། །རང་གི་རྣམ་རྟོག་ཡོད་ཀྱི་བར། །བགེགས་ཀྱི་མཐའ་ལ་ཟད་པ་མེད། །དངོས་སུ་མེད་ཕྱིར་འུབྱུལ་པའི་བློ། །མུ་སྟེགས་ཅན་ལ་འདྲེ་ཡོད་ཕྱིར། །འདྲེར་འཛིན་སྐྱེ་བོ་མ་རིག་པ། །མོ་མའ་རྫུན་གྱིས་གེན་པ་
བསླུས། །དམ་པ་ཀུན་གྱིས་འདྲེ་སུན་བཏོན། །ཡུལ་དང་གོལ་ས་ཆོད་གྱུར་ན། །འདྲེ་མེད་རིག་པ་ཆོས་སྐུར་ཤེས། །ཏཞི་ལ་ལྷ་འདྲེ་མེད་རྟོགས་པས། །ཕན་གནོད་བྱ་བ་མིང་ཡང་མེད། །དེ་ཡང་འདུས་པའི་སྙིང་པོ་
ཉོན། །དེ་དག་ཡེ་ཤེས་རང་ཤར་བ། ༢1། ན་བརྟགས་ཟད་ན་འདྲེ་རྣམས་སྟོངས། །འདྲེ་ནི་ཉིད་འཁྲུལ་ཡིན་པའི་སྲུར། །ར་བོང་རྭ་ལ་ཡོད་རྟོག་འདྲ། །རྫོགས་པ་ཆེན་པོ་ངེས་དོན་ཐམས་ཅད་འདུས་པའི་ཡང་སྙིང་ཀུན་ཏུ་
བཟང་པོ་ཡེ་ཤེས་ཀློང་གི་རྒྱུད། རིན་པ་ཆེ་གརླུར་གྱི་ཡང་ཞུན་ལས། ཡེ་ཤེས་རང་ཤར་གྱི་སྤྱོད་པ་བསྟན་པའི་ལེའུ་སྟེ་སོ་བཞི་པའོ།། །།དེ་ནས་འདུས་པའི་ཚོགས་རྣམས་ལ། །ཀུན་རིག་སྟོན་པས་ཡང་གསུངས་
བགྲོད
 '''
    
    import pprint
    sample = 'འཁོར་དང། འབྱེད་པ མི་མངའ་བའི་རང་བཞིན་གྱིས་ཐབས་ཅིག པར་བཞུགསསོ།  ཧཱུྃ །དེནས་བྱངཆུབ་ཀྱིསེམསཀུན་བྱེད་རྒྱལཔོདེས་འཁོར་ཐམསཅད་ཉིད་ཀྱིརང་བཞིན་དུབྱིན་གྱིསབརླབསཔའིཕྱིར'
    
    s = TextScores(sample, multiline=False)
    print(s)
#    for k in s.__dict__.keys():
#        print k, getattr(s, k)
    for i in  s.invalid: print(i, 'invalid')
#    pprint.pprint(get_text_scores(sample, multiline=True))
#    print get_text_scores(sample)
