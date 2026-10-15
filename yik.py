#! /usr/bin/python
# encoding: utf-8
'''Useful sets of characters and symbols
'''

alphabet = ('ཀ', 'ཁ', 'ག', 'ང', 
                    'ཅ', 'ཆ', 'ཇ', 'ཉ', 
                    'ཏ', 'ཐ', 'ད', 'ན', 
                    'པ', 'ཕ', 'བ', 'མ', 
                    'ཙ', 'ཚ', 'ཛ', 'ཝ', 
                    'ཞ', 'ཟ', 'འ', 'ཡ', 
                    'ར', 'ལ', 'ཤ', 'ས',
                    'ཧ', 'ཨ')

prefixes = ("ག", "ད", "བ", "མ", "འ")

head_letter = ("ར", "ལ", "ས")

root_only = frozenset(('ཀ', 'ཁ', 'ཅ', 'ཆ', 'ཇ', 'ཉ', 'ཏ', 'ཐ', 'པ', 'ཕ',
                     'ཙ', 'ཚ', 'ཛ', 'ཝ', 'ཞ', 'ཟ', 'ཡ', 'ཤ',))
                     #  u'ས' is listed as one technically

subjoined_consonants = frozenset(('ྐ', 'ྑ', 'ྒ', 'ྔ', 'ྕ', 'ྖ', 'ྗ', 'ྙ', 
                                        "ྚ", 'ྟ', 'ྠ', 'ྡ', "ྜ", 'ྣ', 'ྤ', 'ྦ',
                                        'ྥ', 'ྨ', 'ྩ', 'ྪ', 'ྫ', 'ྯ', 'ྮ', 'ྴ', 'ྷ', 'ྻ', 'ྼ'))


subjoined = ('ྱ', 'ྲ', 'ླ', 'ྭ') # wazur is being treated as an official member, for now at least 

suffixes = ('ག', 'ང', 'ད', 'ན', 'བ', 'མ', 'འ', 'ར', 'ལ', 'ས')
second_suffix = ('ས', 'ད')
vowels1 = ('ི', 'ུ', 'ེ', 'ོ')
vowels2 = ('ཨི', 'ཨུ', 'ཨེ', 'ཨོ')
punctuation = ('་', '༌', '།', '༎', '༏', '༑', '༔', '༴')

#~ standard_group = ''
#~ for l in (alphabet, subjoined_consonants, subjoined, vowels1, vowels2, punctuation):
    #~ standard_group += u''.join(l)

#special_chars = ()

# mgo can combinations
twelve_ra_mgo = ('རྐ', 'རྒ', 'རྔ', 'རྗ', 'རྙ', 'རྟ', 'རྡ', 'རྣ',
                             'རྦ', 'རྨ', 'རྩ', 'རྫ')
ten_la_mgo = ('ལྐ', 'ལྒ', 'ལྔ', 'ལྕ', 'ལྗ', 'ལྟ', 'ལྡ', 'ལྤ',
                        'ལྦ', 'ལྷ')
eleven_sa_mgo = ('སྐ', 'སྒ', 'སྔ', 'སྙ', 'སྟ', 'སྡ', 'སྣ', 'སྤ',
                                'སྦ', 'སྨ', 'སྩ')

# 'dogs can combinations
seven_ya_tags = ('ཀྱ', 'ཁྱ', 'གྱ', 'པྱ', 'ཕྱ', 'བྱ', 'མྱ')
twelve_ra_tags = ('ཀྲ', 'ཁྲ', 'གྲ', 'ཏྲ', 'ཐྲ', 'དྲ', 'པྲ', 'ཕྲ', 'བྲ', 'མྲ', 'ཧྲ', 'སྲ')
six_la_tags = ('ཀླ', 'གླ', 'བླ', 'ཟླ', 'རླ', 'སླ')

# three tiered stacks
ya_tags_stack = ('རྐྱ', 'རྒྱ', 'རྨྱ', 'སྐྱ', 'སྒྱ', 'སྤྱ', 'སྦྱ', 'སྨྱ')
ra_tags_stack = ('སྐྲ', 'སྒྲ', 'སྣྲ', 'སྤྲ', 'སྦྲ', 'སྨྲ')

# grammar 
seven_la_don = ('སུ', 'ར', 'རུ', 'དུ', 'ན', 'ལ', 'ཏུ')
grel_sgra = ('གི', 'ཀྱི', 'གྱི', 'འི', 'ཡི')
byed_sgra = ('གིས', 'ཀྱིས', 'གྱིས', 'འིས', 'ཡིས')
terminating_syllables = ('གོ', 'ངོ', 'དོ', 'ནོ', 'བོ', 'མོ', 'འོ', 'རོ', 'ལོ', 'སོ')
rgyan_sdud = ('ཀྱང','འང','ཡང')
num = ('༡', '༢', '༣', '༤', '༥', '༦', '༧', '༨', '༩', '༠')

# ambiguous cases
amb1 = ('བགས', 'མངས')
amb2 = ('དགས', 'འགས', 'དབས', 'དམས')

    
letters = ('\\u0f40', '\\u0f41', '\\u0f42', '\\u0f43', '\\u0f44', '\\u0f45',
    '\\u0f46', '\\u0f47', '\\u0f49', '\\u0f4a', '\\u0f4b', '\\u0f4c', '\\u0f4d',
    '\\u0f4e', '\\u0f4f', '\\u0f50', '\\u0f51', '\\u0f52', '\\u0f53', '\\u0f54',
    '\\u0f55', '\\u0f56', '\\u0f57', '\\u0f58', '\\u0f59', '\\u0f5a', '\\u0f5b',
    '\\u0f5c', '\\u0f5d', '\\u0f5e', '\\u0f5f', '\\u0f60', '\\u0f61', '\\u0f62', 
    '\\u0f63', '\\u0f64', '\\u0f65', '\\u0f66', '\\u0f67', '\\u0f68', '\\u0f69', 
    '\\u0f6a', '\\u0f6b', '\\u0f6c')

subjoined_letters = ('\\u0f90', '\\u0f91', '\\u0f92', '\\u0f93', '\\u0f94',
    '\\u0f95', '\\u0f96', '\\u0f97', '\\u0f99', '\\u0f9a', '\\u0f9b', '\\u0f9c',
    '\\u0f9d', '\\u0f9e', '\\u0f9f', '\\u0fa0', '\\u0fa1', '\\u0fa2', '\\u0fa3',
    '\\u0fa4', '\\u0fa5', '\\u0fa6', '\\u0fa7', '\\u0fa8', '\\u0fa9', '\\u0faa',
    '\\u0fab', '\\u0fac', '\\u0fad', '\\u0fae', '\\u0faf', '\\u0fb0', '\\u0fb1',
    '\\u0fb2', '\\u0fb3', '\\u0fb4', '\\u0fb5', '\\u0fb6', '\\u0fb7', '\\u0fb8',
    '\\u0fb9', '\\u0fba', '\\u0fbb', '\\u0fbc')


f_vowels = ('\\u0f71', '\\u0f72', '\\u0f73', '\\u0f74', '\\u0f75', '\\u0f76',
     '\\u0f77', '\\u0f78', '\\u0f79', '\\u0f7a', '\\u0f7b', '\\u0f7c', '\\u0f7d', 
     '\\u0f80', '\\u0f81')

signs = ('\\u0f1a', '\\u0f1b', '\\u0f1c', '\\u0f1d', '\\u0f1e', '\\u0f1f',
    '\\u0f3e', '\\u0f3f', '\\u0f7e', '\\u0f7f', '\\u0f82', '\\u0f83', '\\u0f86',
    '\\u0f87', '\\u0f88', '\\u0f89', '\\u0f8a', '\\u0f8b', '\\u0fc0', '\\u0fc1',
    '\\u0fc2', '\\u0fc3', '\\u0fce', '\\u0fcf') # does not include logotypes or astrological signs

marks = ('\\u0f01', '\\u0f02', '\\u0f03', '\\u0f04', '\\u0f05', '\\u0f06',
    '\\u0f07', '\\u0f08', '\\u0f09', '\\u0f0a', '\\u0f0b', '\\u0f0c', '\\u0f0d',
    '\\u0f0e', '\\u0f0f', '\\u0f10', '\\u0f11', '\\u0f12', '\\u0f13', '\\u0f14',
    '\\u0f34', '\\u0f35', '\\u0f36', '\\u0f37', '\\u0f38', '\\u0f39', '\\u0f3a',
    '\\u0f3b', '\\u0f3c', '\\u0f3d', '\\u0f84', '\\u0f85', '\\u0fd0', '\\u0fd1', 
    '\\u0fd2', '\\u0fd3', '\\u0fd4')

shad = ('\\u0f06', '\\u0f07', '\\u0f08', '\\u0f0d', '\\u0f0e', '\\u0f0f', '\\u0f10', '\\u0f11', '\\u0f12')

syllables = ('\\u0f00',)
logotype = ('\\u0f15', '\\u0f16')
astro_sign = ('\\u0f17', '\\u0f18', '\\u0f19')
digit = ('\\u0f20', '\\u0f21', '\\u0f22', '\\u0f23', '\\u0f24', '\\u0f25', '\\u0f26',
    '\\u0f27', '\\u0f28', '\\u0f29', '\\u0f2a', '\\u0f2b', '\\u0f2c', '\\u0f2d',
    '\\u0f2e', '\\u0f2f', '\\u0f30', '\\u0f31', '\\u0f32', '\\u0f33')

symbol = ('\\u0fc4', '\\u0fc5', '\\u0fc6', '\\u0fc7', '\\u0fc8', '\\u0fc9',
    '\\u0fca', '\\u0fcb', '\\u0fcc')

norm_roots = {
    '\\u0f41': '\\u0f41', '\\u0f40': '\\u0f40', '\\u0f43': '\\u0f43', 
    '\\u0f42': '\\u0f42', '\\u0f45': '\\u0f45', '\\u0f44': '\\u0f44', 
    '\\u0f47': '\\u0f47', '\\u0f46': '\\u0f46', '\\u0f49': '\\u0f49', 
    '\\u0f4b': '\\u0f4b', '\\u0f4a': '\\u0f4a', '\\u0f4d': '\\u0f4d', 
    '\\u0f4c': '\\u0f4c', '\\u0f4f': '\\u0f4f', '\\u0f4e': '\\u0f4e', 
    '\\u0f51': '\\u0f51', '\\u0f50': '\\u0f50', '\\u0f53': '\\u0f53', 
    '\\u0f52': '\\u0f52', '\\u0f55': '\\u0f55', '\\u0f54': '\\u0f54', 
    '\\u0f57': '\\u0f57', '\\u0f56': '\\u0f56', '\\u0f59': '\\u0f59', 
    '\\u0f58': '\\u0f58', '\\u0f5b': '\\u0f5b', '\\u0f5a': '\\u0f5a', 
    '\\u0f5d': '\\u0f5d', '\\u0f5c': '\\u0f5c', '\\u0f5f': '\\u0f5f', 
    '\\u0f5e': '\\u0f5e', '\\u0f61': '\\u0f61', '\\u0f60': '\\u0f60', 
    '\\u0f63': '\\u0f63', '\\u0f62': '\\u0f62', '\\u0f65': '\\u0f65', 
    '\\u0f64': '\\u0f64', '\\u0f67': '\\u0f67', '\\u0f66': '\\u0f66', 
    '\\u0f69': '\\u0f69', '\\u0f68': '\\u0f68', '\\u0f6b': '\\u0f6b', 
    '\\u0f6a': '\\u0f6a', '\\u0f6c': '\\u0f6c',
    '\\u0f91': '\\u0f41', '\\u0f90': '\\u0f40', '\\u0f93': '\\u0f43', 
    '\\u0f92': '\\u0f42', '\\u0f95': '\\u0f45', '\\u0f94': '\\u0f44', 
    '\\u0f97': '\\u0f47', '\\u0f96': '\\u0f46', '\\u0f99': '\\u0f49', 
    '\\u0f9b': '\\u0f4b', '\\u0f9a': '\\u0f4a', '\\u0f9d': '\\u0f4d', 
    '\\u0f9c': '\\u0f4c', '\\u0f9f': '\\u0f4f', '\\u0f9e': '\\u0f4e', 
    '\\u0fa1': '\\u0f51', '\\u0fa0': '\\u0f50', '\\u0fa3': '\\u0f53', 
    '\\u0fa2': '\\u0f52',     '\\u0fa5': '\\u0f55', '\\u0fa4': '\\u0f54', 
    '\\u0fa7': '\\u0f57', '\\u0fa6': '\\u0f56',     '\\u0fa9': '\\u0f59', 
    '\\u0fa8': '\\u0f58', '\\u0fab': '\\u0f5b', '\\u0faa': '\\u0f5a',
    '\\u0fad': '\\u0f5d', '\\u0fac': '\\u0f5c', '\\u0faf': '\\u0f5f', 
    '\\u0fae': '\\u0f5e',     '\\u0fb1': '\\u0f61', '\\u0fb0': '\\u0f60', 
    '\\u0fb3': '\\u0f63', '\\u0fb2': '\\u0f62', '\\u0fb5': '\\u0f65', 
    '\\u0fb4': '\\u0f64', '\\u0fb7': '\\u0f67', '\\u0fb6': '\\u0f66', 
    '\\u0fb9': '\\u0f69', '\\u0fb8': '\\u0f68', '\\u0fbb': '\\u0f6b', 
    '\\u0fba': '\\u0f6a',     '\\u0fbc': '\\u0f6c'
    }

word_parts = letters + subjoined_letters + f_vowels
non_letters = signs + syllables + marks + logotype + astro_sign + digit + symbol + ("\n", "\r", " ", "\t", "\\u00A0")
non_letters2 = signs + syllables + marks + logotype + astro_sign + digit + symbol

word_parts_set = frozenset(word_parts)
non_letters_set = frozenset(non_letters)


lexical_map = {"root_only":root_only, "subjoined":subjoined, 
                        "subjoined_cons":subjoined_consonants, 
                        "vowel":f_vowels, "non_letter":non_letters}

