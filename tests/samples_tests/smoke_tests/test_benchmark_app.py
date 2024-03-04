"""
 Copyright (C) 2018-2023 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
import os
import pytest
import logging as log
import sys
from common.samples_common_test_class import SamplesCommonTestClass
from common.samples_common_test_class import get_tests

log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)

test_data_fp32_async = get_tests \
    (cmd_params={'i': [os.path.join('227x227', 'dog.bmp')],
                 'm': [os.path.join('squeezenet1.1', 'FP32', 'squeezenet1.1.xml')],
                 'batch': [1],
                 'sample_type': ['C++', 'Python'],
                 'd': ['CPU'],
                 'api': ['async'],
                 'nireq': ['4'],
                 'niter': ['10'], },
     use_device=['d']
     )

test_data_fp32_sync = get_tests \
    (cmd_params={'i': [os.path.join('227x227', 'dog.bmp')],
                 'm': [os.path.join('squeezenet1.1', 'FP32', 'squeezenet1.1.xml')],
                 'batch': [1],
                 'sample_type': ['C++', 'Python'],
                 'd': ['CPU'],
                 'niter': ['10'],
                 'api': ['sync']},
     use_device=['d']
     )



class TestBenchmarkApp(SamplesCommonTestClass):
    @classmethod
    def setup_class(cls):
        cls.sample_name = 'benchmark_app'
        super().setup_class()

    @pytest.mark.parametrize("param", test_data_fp32_async)
    @pytest.mark.skip("Ticket: 106850")
    def test_benchmark_app_sample_fp32_async(self, param):
        _check_output(self, param)

    @pytest.mark.parametrize("param", test_data_fp32_sync)
    @pytest.mark.skip("Ticket: 106850")
    def test_benchmark_app_fp32_sync(self, param):
        _check_output(self, param)

def stream_checker(param, config_json):
    device = param.get('d', 'CPU')
    if device not in config_json.keys():
        log.error("device not found")
        return False
    if 'NUM_STREAMS' not in config_json[device].keys():
        log.error("NUM_STREAMS not found")
        return False
    if param['nstreams'] == config_json[device]['NUM_STREAMS']:
        return True
    else:
        log.error("value of nstreams is false")
        return False

def pin_checker(param, config_json):
    device = param.get('d', 'CPU')
    if device not in config_json.keys():
        log.error("device not found")
        return False
    if 'AFFINITY' not in config_json[device].keys():
        log.error("AFFINITY not found")
        return False
    if param['pin'] == 'YES' and config_json[device]['AFFINITY'] == 'CORE':
        return True
    elif param['pin'] == 'NO' and config_json[device]['AFFINITY'] == 'NONE':
        return True
    elif param['pin'] == config_json[device]['AFFINITY']:
        return True
    else:
        log.error("value of pin is false")
        return False
    
def _check_output(self, param):
    """
    Benchmark_app has functional and accuracy testing.
    For accuracy the test checks if 'FPS' 'Latency' in output. If both exist - the est passed
    """

    # Run _test function, that returns stdout or 0.
    # 0 is for device FPGA, Myriad, HDDL - skip accuraccy check on these devices
    if 'dump_config' not in param:
        param['dump_config'] = os.path.join(os.environ.get('WORKSPACE'), 'config.json')
    stdout = self._test(param)
    print(stdout)
    config_file_name = param['dump_config']
    config = open(config_file_name)
    lines = config.readlines()
    print('config file name:', param['dump_config'])
    for line in lines:
        print(line)
    config.seek(0, 0)
    if not config:
        return 0
    config_json = json.load(config) 
    config.close()
    if not stdout:
        return 0
    stdout = stdout.split('\n')
    is_ok = False
    flag = 0
    for line in stdout:
        if 'FPS' in line:
            is_ok = True
    if is_ok == False:
        flag = 1
        log.error("No FPS in output")
    assert flag == 0, "Wrong output of FPS"

    flag = 0
    is_ok = False
    for line in stdout:
        if 'Latency' in line:
            is_ok = True
    if is_ok == False:
        flag = 1
        log.error("No Latency in output")
    assert flag == 0, "Wrong output of Latency"

    is_ok = False
    if 'nstreams' in param:
        is_ok = stream_checker(param, config_json)
        if is_ok == False:
            log.error("No expected nstreams in output")
            assert False, "check nstreams failed"
    is_ok = False
    if 'pin' in param:
        is_ok = pin_checker(param, config_json)
        if is_ok == False:
            log.error("No expected pin in output")
            assert False, "check pin failed"

    log.info('Accuracy passed')
