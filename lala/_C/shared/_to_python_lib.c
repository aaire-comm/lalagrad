#include <python3.12/Python.h>

PyObject* _to_python_list(PyObject *self, PyObject *args){

    int64_t arr;
    PyObject *shape, *strides;
    int off;

    if(!PyArg_ParseTuple(args, "iooi", &arr, &shape, &strides, &off))
        return NULL;


    printg("%d,");
    return Py_None;
}