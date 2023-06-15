# printarr
A small python utility to pretty-print a table summarizing arrays &amp; scalars from numpy, pytorch, etc.

### Example

Calling `printarr(my_arr1, my_arr2, ...)` prints a table like:

```
       name | dtype         | shape         | type          | device | min         | max         | mean       
--------------------------------------------------------------------------------------------------------------
     [None] | None          | N/A           | NoneType      |        | N/A         | N/A         | N/A        
    intval1 | int           | scalar        | int           |        |      7      |      7      |      7     
    intval2 | int           | scalar        | int           |        |     -3      |     -3      |     -3     
  floatval0 | float         | scalar        | float         |        |     42      |     42      |     42     
  floatval1 | float         | scalar        | float         |        | 5.5e-12     | 5.5e-12     | 5.5e-12    
  floatval2 | float         | scalar        | float         |        | 7.72324e+44 | 7.72324e+44 | 7.72324e+44
     npval1 | int64         | [100]         | numpy.ndarray |        |      0      |     99      |   49.5     
     npval2 | int64         | [10000]       | numpy.ndarray |        |      0      |   9999      | 4999.5     
     npval3 | uint64        | [10000]       | numpy.ndarray |        |      0      |   9999      | 4999.5     
     npval4 | float32       | [100, 10, 10] | numpy.ndarray |        |      0      |   9999      | 4999.5     
[temporary] | float32       | [10, 8]       | numpy.ndarray |        |      2      |     99      |   50.5     
     npval5 | int64         | []            | numpy.int64   |        |   9999      |   9999      |   9999     
  torchval1 | torch.float32 | [1000, 12, 3] | torch.Tensor  | cpu    | -4.08445    | 3.90982     | 0.00404567 
  torchval2 | torch.float32 | [1000, 12, 3] | torch.Tensor  | cuda:0 | -3.87309    | 3.90342     | 0.00339224 
  torchval3 | torch.int64   | [1000]        | torch.Tensor  | cpu    |      0      |    999      | N/A        
  torchval4 | torch.int64   | []            | torch.Tensor  | cpu    |      0      |      0      | N/A
```

### Installation

`pip install printarr`

`from printarr import printarr`

### Docs

The package exposes a single function called `printarr()`. Call it like: `printarr(my_arr, some_other_arr, maybe_a_scalar)`.

The function accepts a variable number of arguments. Arrays can also be passed as named optional arguments.

Inputs can be:
- Numpy tensor arrays
- Pytorch tensor arrays
- Jax tensor arrays
- Python ints / floats
- `None`
- It may also work with other array-like types, but they have not been tested.
- Input values which are not arrays or numeric types (strings, objects, etc) will be printed as blank rows in the table.

When arrays are passed as variable arguments, the printed name of the array in the table is inferred from the variable name in the outer scope, when possible. When arrays are passed as named keyword arguments, the key name is used.

Pass an integer for the `printarr_float_width` option to specify the precision to which floating point types are printed.

Author: Nicholas Sharp (nmwsharp.com)


