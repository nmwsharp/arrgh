import inspect

def arrgh(*arrs, arrgh_float_width=6, **kwargs):
    """
    Print a pretty table giving name, shape, dtype, type, and content information for input tensors or scalars.

    Call like: arrgh(my_arr, some_other_arr, maybe_a_scalar). Accepts a variable number of arguments.

    Arrays can also be passed as named optional arguments.

    Inputs can be:
        - Numpy tensor arrays
        - Pytorch tensor arrays
        - Jax tensor arrays
        - Python ints / floats
        - None

    It may also work with other array-like types, but they have not been tested.

    Input values which are not arrays or numeric types (strings, objects, etc) will be printed as blank rows in the table.

    When arrays are passed as variable arguments, the printed name of the array in the table is inferred from
    the variable name in the outer scope, when possible. When arrays are passed as named keyword arguments, the
    key name is used.

    Pass an integer for the `arrgh_float_width` option specify the precision to which floating point types are printed.

    Author: Nicholas Sharp (nmwsharp.com)
    Canonical source: https://github.com/nmwsharp/arrgh, and the pypi package `arrgh`
    License: MIT license, and it is also released into the public domain. 
             Please retain this docstring as a reference.
    """
    
    frame = inspect.currentframe().f_back
    default_name = "[temporary]"

    ## helpers to gather data about each array
    def name_from_outer_scope(a):
        if a is None:
            return '[None]'
        name = default_name
        for k, v in frame.f_locals.items():
            if v is a:
                name = k
                break
        return name
    def dtype_str(a):

        if a is None:
            return 'None'
        if isinstance(a, int):
            return 'int'
        if isinstance(a, float):
            return 'float'

        # try to read the dtype, but if it fails
        # just retrn 'invalid'
        outval = 'invalid'
        try: outval = str(a.dtype)
        except: pass

        return outval
    def shape_str(a):
        if a is None:
            return 'N/A'
        if isinstance(a, int):
            return 'scalar'
        if isinstance(a, float):
            return 'scalar'
        return str(list(a.shape))
    def type_str(a):
        return str(type(a))[8:-2] # TODO this is is weird... what's the better way?
    def device_str(a):
        if hasattr(a, 'device'):
            device_str = str(a.device)
            if len(device_str) < 10:
                # heuristic: jax returns some goofy long string we don't want, ignore it
                return device_str
        return ""
    def format_float(x):
        return f"{x:.{arrgh_float_width}g}"
    def minmaxmean_str(a):
        if a is None:
            return ('N/A', 'N/A', 'N/A')
        if isinstance(a, int) or isinstance(a, float): 
            return (format_float(a), format_float(a), format_float(a))

        # compute min/max/mean. if anything goes wrong, just print 'N/A'
        min_str = "N/A"
        try: min_str = format_float(a.min())
        except: pass
        max_str = "N/A"
        try: max_str = format_float(a.max())
        except: pass
        mean_str = "N/A"
        try: mean_str = format_float(a.mean())
        except: pass

        return (min_str, max_str, mean_str)
    


    try:

        props = ['name', 'dtype', 'shape', 'type', 'device', 'min', 'max', 'mean']

        # precompute all of the properties for each input
        str_props = []
    
        def add_props(a, name):
            dtype_str_val = dtype_str(a)
            is_valid = dtype_str_val != 'invalid'
            if is_valid:
                minmaxmean = minmaxmean_str(a)
                str_props.append({
                    'name' : name,
                    'dtype' : dtype_str_val,
                    'shape' : shape_str(a),
                    'type' : type_str(a),
                    'device' : device_str(a),
                    'min' : minmaxmean[0],
                    'max' : minmaxmean[1],
                    'mean' : minmaxmean[2],
                })
            else: # invalid entries
                str_props.append({
                    'name' : name,
                    'dtype' : dtype_str_val,
                    'shape' : '',
                    'type' : type_str(a),
                    'device' : '',
                    'min' : '',
                    'max' : '',
                    'mean' : '',
                })

        # variadic args first, with names automatically inferred from scope
        for a in arrs:
            add_props(a, name_from_outer_scope(a))
        
        # keyworkd args next, taking the name from the keyword
        for key in kwargs:
            a = kwargs[key]
            add_props(a, key)

        # for each property, compute its length
        maxlen = {}
        for p in props: maxlen[p] = 0
        for sp in str_props:
            for p in props:
                maxlen[p] = max(maxlen[p], len(sp[p]))

        # if any property got all empty strings, don't bother printing it, remove if from the list
        props = [p for p in props if maxlen[p] > 0]

        # handle width in the case where the property name is longer than any of the entries
        maxlen = {p: max(maxlen[p], len(p)) for p in props}

        # print a header
        header_str = ""
        for p in props:
            prefix =  "" if p == 'name' else " | "
            fmt_key = ">" if p == 'name' else "<"
            header_str += f"{prefix}{p:{fmt_key}{maxlen[p]}}"
        print(header_str)
        print("-"*len(header_str))
            
        # now print the acual arrays
        for strp in str_props:
            for p in props:
                prefix =  "" if p == 'name' else " | "
                fmt_key = ">" if p == 'name' else "<"
                print(f"{prefix}{strp[p]:{fmt_key}{maxlen[p]}}", end='')
            print("")

    finally:
        del frame
