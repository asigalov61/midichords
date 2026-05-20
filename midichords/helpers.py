#! /usr/bin/python3

r'''###############################################################################
###################################################################################
#
#	Helpers Python Module
#	Version 1.0
#
#	Project Los Angeles
#
#	Tegridy Code 2026
#
#   https://github.com/Tegridy-Code/Project-Los-Angeles
#
###################################################################################
###################################################################################
#
#   Copyright 2026 Project Los Angeles / Tegridy Code
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
###################################################################################
'''

print('=' * 70)
print('Loading midichords helpers module...')
print('Please wait...')
print('=' * 70)

__version__ = '1.0.0'

print('midichords helpers module version', __version__)
print('=' * 70)

###################################################################################

import importlib.resources as pkg_resources

from . import models

from typing import List, Dict

###################################################################################

def get_package_models() -> List[Dict]:
    
    """
    Get models included with midisim package
    
    Returns
    -------
    List of dicts: {'model': model_file_name,
                    'path': model_full_path
                   }
    """
    
    models_dict = []
    
    for resource in pkg_resources.contents(models):
        if resource.endswith('.pth'):
            with pkg_resources.path(models, resource) as p:
                mdic = {'model': resource,
                        'path': str(p)
                       }
                
                models_dict.append(mdic)
                
    return sorted(models_dict, key=lambda x: x['model'])

###################################################################################

print('Module is loaded!')
print('Enjoy! :)')
print('=' * 70)

###################################################################################
# This is the end of the Helpers Python Module
###################################################################################