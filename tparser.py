# encoding: utf-8

try:
    from . import yik
except ImportError:
    import yik
import codecs
import re

test_str = '''ད་རིང་བོད་ཀྱི་ས་ཁུལ་ཨ་མདོ་འཛམ་ཐང་ནང་རང་ལོ་་་ ༣༠ ཡིན་པའི་སྐྱེད་མ་གཅིག་གིས་རང་ལུས་མེར་བསྲེགས་བཏང་ནས་རྒྱ་ནག་གཞུང་ལ་ངོ་རྒོལ་གནང་རྗེས་འདས་གྲོངས་སོང་འདུག་པ་དང་། 
བུ་ཕྲུག་བཞིའི་ཨ་མ་ཡིན་པ་སྐལ་སྐྱིད་ལགས་ཀྱིས་ད་རིང་བོད་ཀྱི་དུས་ཚོད་ལྟར་བྱས་ན་ཉིན་རྒྱབ་ཚོུད་ ༣ དང་སྐར་མ་སུམ་ཅུ་ཙམ་ལ་འཛམ་ཐང་ཇོ་ནང་དགོན་ཆེན་གྱི་ཉེ་འགྲམ་དུ་རང་ལུས་མེ་ལ་བསྲེགས་'''    

def load_normalize_ocr(flname):
    tiffname_start = 'tbocrtifs'
    page_ignore = '༄༅། །'
    
    contents = codecs.open(flname, 'r', 'utf-8').readlines()
    contents = [l for l in contents if not l.startswith(tiffname_start)]
    contents = ''.join(contents)
    contents = contents.replace(page_ignore, '')
    contents = contents.replace('༄༅', '')
    
    contents = re.sub(r'[་]{2,}\n', '་\n', contents) 
    contents = contents.replace('\n', '')
    contents = re.sub(r'^This.*corrections\.', '', contents) 

    return contents

def parse_stacks(tstr):
    '''Convert a string to a list of stacks, where a stack is defined as a
    column of letters within a syllable. Run time is linear'''
    
    all_stacks = []
    current_stack = []
    for i in tstr:
        if i in yik.letters or i not in yik.word_parts:
            if not current_stack:
                current_stack.append(i)
            else:
                all_stacks.append(''.join(current_stack))
                current_stack = [i]
        else:
            current_stack.append(i)
    all_stacks.append(''.join(current_stack))
    return all_stacks
 
def parse_syllables(tstr, omit_tsek=True): 
    all_syllables = []
    current_syll = []
    for i in tstr:
        if i in yik.word_parts_set:
            current_syll.append(i)
        else:
            if current_syll:
                all_syllables.append(''.join(current_syll))
                current_syll = []
            if i == '་' and omit_tsek: 
                    continue
            else:
                all_syllables.append(i)
                current_syll = []
    all_syllables.append(''.join(current_syll))
    return all_syllables
            
    
if __name__ == '__main__':
    for stack in parse_stacks('བདགརབ'):
        print(stack, end=' ')

    for syl in parse_syllables(test_str, omit_tsek=False):
        print(syl)
