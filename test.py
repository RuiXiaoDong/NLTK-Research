'''
Created on Aug 24, 2015

@author: sdxr
'''
import re
originalstring = 'look d###$$$own$$$### ###$$$own$$$###ed'
pattern_prefix = r'[a-z]+###\$\$\$[^(###\$\$\$)]+\$\$\$###'
pattern_surfix = r'###\$\$\$[^(###\$\$\$)]+\$\$\$###[a-z]+'
for pattern_match in re.findall(pattern_prefix, originalstring):
    pattern_replace = pattern_match.replace('###$$$','')
    pattern_replace = pattern_replace.replace('$$$###','')
    originalstring = originalstring.replace(pattern_match, pattern_replace)
    print originalstring
    
for pattern_match in re.findall(pattern_surfix, originalstring):
    pattern_replace = pattern_match.replace('###$$$','')
    pattern_replace = pattern_replace.replace('$$$###','')
    originalstring = originalstring.replace(pattern_match, pattern_replace)
    print originalstring