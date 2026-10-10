import multiprocessing

def _run_exec(full_code, queue):
    import math
    from typing import List, Dict, Tuple, Set, Optional, Union, Any
    import collections
    import itertools
    import re
    import hashlib
    
    exec_globals = {
        'math': math, 'List': List, 'Dict': Dict, 'Tuple': Tuple, 'Set': Set,
        'Optional': Optional, 'Union': Union, 'Any': Any,
        'collections': collections, 'itertools': itertools, 're': re,
        'hashlib': hashlib
    }
    
    def assertion(a, b, cmp=0):
        if cmp == 0:
            assert a == b, f"{a} != {b}"
        else:
            assert abs(a - b) <= cmp, f"{a} != {b} within {cmp}"
    exec_globals['assertion'] = assertion
    
    try:
        exec(full_code, exec_globals)
        queue.put((True, "Passed"))
    except AssertionError as e:
        queue.put((False, "Assertion Error (Edge Case Failed)"))
    except Exception as e:
        queue.put((False, f"Execution Error: {type(e).__name__}"))
