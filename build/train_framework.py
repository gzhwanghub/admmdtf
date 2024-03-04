# This file was automatically generated by SWIG (http://www.swig.org).
# Version 4.0.2
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.

from sys import version_info as _swig_python_version_info
if _swig_python_version_info < (2, 7, 0):
    raise RuntimeError("Python 2.7 or later required")

# Import the low-level C/C++ module
if __package__ or "." in __name__:
    from . import _train_framework
else:
    import _train_framework

try:
    import builtins as __builtin__
except ImportError:
    import __builtin__

def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except __builtin__.Exception:
        strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)


def _swig_setattr_nondynamic_instance_variable(set):
    def set_instance_attr(self, name, value):
        if name == "thisown":
            self.this.own(value)
        elif name == "this":
            set(self, name, value)
        elif hasattr(self, name) and isinstance(getattr(type(self), name), property):
            set(self, name, value)
        else:
            raise AttributeError("You cannot add instance attributes to %s" % self)
    return set_instance_attr


def _swig_setattr_nondynamic_class_variable(set):
    def set_class_attr(cls, name, value):
        if hasattr(cls, name) and not isinstance(getattr(cls, name), property):
            set(cls, name, value)
        else:
            raise AttributeError("You cannot add class attributes to %s" % cls)
    return set_class_attr


def _swig_add_metaclass(metaclass):
    """Class decorator for adding a metaclass to a SWIG wrapped class - a slimmed down version of six.add_metaclass"""
    def wrapper(cls):
        return metaclass(cls.__name__, cls.__bases__, cls.__dict__.copy())
    return wrapper


class _SwigNonDynamicMeta(type):
    """Meta class to enforce nondynamic attributes (no new attributes) for a class"""
    __setattr__ = _swig_setattr_nondynamic_class_variable(type.__setattr__)


class SparseFeature(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    index = property(_train_framework.SparseFeature_index_get, _train_framework.SparseFeature_index_set)
    numUniqueFeatures = property(_train_framework.SparseFeature_numUniqueFeatures_get, _train_framework.SparseFeature_numUniqueFeatures_set)
    featureIndex = property(_train_framework.SparseFeature_featureIndex_get, _train_framework.SparseFeature_featureIndex_set)
    featureVec = property(_train_framework.SparseFeature_featureVec_get, _train_framework.SparseFeature_featureVec_set)
    numFeatures = property(_train_framework.SparseFeature_numFeatures_get, _train_framework.SparseFeature_numFeatures_set)

    def __init__(self):
        _train_framework.SparseFeature_swiginit(self, _train_framework.new_SparseFeature())
    __swig_destroy__ = _train_framework.delete_SparseFeature

# Register SparseFeature in _train_framework:
_train_framework.SparseFeature_swigregister(SparseFeature)

class DenseFeature(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    index = property(_train_framework.DenseFeature_index_get, _train_framework.DenseFeature_index_set)
    featureVec = property(_train_framework.DenseFeature_featureVec_get, _train_framework.DenseFeature_featureVec_set)
    numFeatures = property(_train_framework.DenseFeature_numFeatures_get, _train_framework.DenseFeature_numFeatures_set)

    def __init__(self):
        _train_framework.DenseFeature_swiginit(self, _train_framework.new_DenseFeature())
    __swig_destroy__ = _train_framework.delete_DenseFeature

# Register DenseFeature in _train_framework:
_train_framework.DenseFeature_swigregister(DenseFeature)

class Matrix(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, *args):
        _train_framework.Matrix_swiginit(self, _train_framework.new_Matrix(*args))

    def __call__(self, *args):
        return _train_framework.Matrix___call__(self, *args)

    def push_back(self, v):
        return _train_framework.Matrix_push_back(self, v)

    def remove(self, i):
        return _train_framework.Matrix_remove(self, i)

    def numRows(self):
        return _train_framework.Matrix_numRows(self)

    def numColumns(self):
        return _train_framework.Matrix_numColumns(self)

    def size(self):
        return _train_framework.Matrix_size(self)
    __swig_destroy__ = _train_framework.delete_Matrix

# Register Matrix in _train_framework:
_train_framework.Matrix_swigregister(Matrix)


def readFeatureLabelsLibSVM(fname, features, y, n, numFeatures):
    return _train_framework.readFeatureLabelsLibSVM(fname, features, y, n, numFeatures)

def sum(x):
    return _train_framework.sum(x)

def vectorAddition(x, y, z):
    return _train_framework.vectorAddition(x, y, z)

def vectorFeatureAddition(*args):
    return _train_framework.vectorFeatureAddition(*args)

def vectorScalarAddition(x, a, z):
    return _train_framework.vectorScalarAddition(x, a, z)

def vectorSubtraction(x, y, z):
    return _train_framework.vectorSubtraction(x, y, z)

def vectorFeatureSubtraction(*args):
    return _train_framework.vectorFeatureSubtraction(*args)

def vectorScalarSubtraction(x, a, z):
    return _train_framework.vectorScalarSubtraction(x, a, z)

def elementMultiplication(*args):
    return _train_framework.elementMultiplication(*args)

def elementPower(*args):
    return _train_framework.elementPower(*args)

def scalarMultiplication(*args):
    return _train_framework.scalarMultiplication(*args)

def innerProduct(x, y):
    return _train_framework.innerProduct(x, y)

def featureProduct(*args):
    return _train_framework.featureProduct(*args)

def featureProductCheck(*args):
    return _train_framework.featureProductCheck(*args)

def outerProduct(x, y, m):
    return _train_framework.outerProduct(x, y, m)

def argMax(x):
    return _train_framework.argMax(x)

def norm(x, type=2):
    return _train_framework.norm(x, type)

def abs(*args):
    return _train_framework.abs(*args)

def sign(*args):
    return _train_framework.sign(*args)

def multiplyAccumulate(*args):
    return _train_framework.multiplyAccumulate(*args)

def __add__(*args):
    return _train_framework.__add__(*args)

def __sub__(*args):
    return _train_framework.__sub__(*args)

def __mul__(*args):
    return _train_framework.__mul__(*args)

def __iadd__(*args):
    return _train_framework.__iadd__(*args)

def __isub__(*args):
    return _train_framework.__isub__(*args)

def __imul__(x, a):
    return _train_framework.__imul__(x, a)

def __eq__(x, y):
    return _train_framework.__eq__(x, y)

def __ne__(x, y):
    return _train_framework.__ne__(x, y)

def __lt__(x, y):
    return _train_framework.__lt__(x, y)

def __le__(x, y):
    return _train_framework.__le__(x, y)

def __gt__(x, y):
    return _train_framework.__gt__(x, y)

def __ge__(x, y):
    return _train_framework.__ge__(x, y)

def __lshift__(os, x):
    return _train_framework.__lshift__(os, x)
class ContinuousFunctions(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    isSmooth = property(_train_framework.ContinuousFunctions_isSmooth_get, _train_framework.ContinuousFunctions_isSmooth_set)

    def __init__(self, *args):
        _train_framework.ContinuousFunctions_swiginit(self, _train_framework.new_ContinuousFunctions(*args))
    __swig_destroy__ = _train_framework.delete_ContinuousFunctions

    def evalGradient(self, x):
        return _train_framework.ContinuousFunctions_evalGradient(self, x)

    def eval(self, *args):
        return _train_framework.ContinuousFunctions_eval(self, *args)

    def evalStochasticGradient(self, x, batch):
        return _train_framework.ContinuousFunctions_evalStochasticGradient(self, x, batch)

    def evalStochastic(self, x, f, g, miniBatch):
        return _train_framework.ContinuousFunctions_evalStochastic(self, x, f, g, miniBatch)

    def evalHessian(self, x):
        return _train_framework.ContinuousFunctions_evalHessian(self, x)

    def evalHessianVectorProduct(self, x, v, Hxv):
        return _train_framework.ContinuousFunctions_evalHessianVectorProduct(self, x, v, Hxv)

    def __call__(self, x):
        return _train_framework.ContinuousFunctions___call__(self, x)

    def size(self):
        return _train_framework.ContinuousFunctions_size(self)

    def length(self):
        return _train_framework.ContinuousFunctions_length(self)

# Register ContinuousFunctions in _train_framework:
_train_framework.ContinuousFunctions_swigregister(ContinuousFunctions)


def gdNesterov(c, x0, alpha=1, gamma=1e-4, maxEval=1000, TOL=1e-3, resetAlpha=True, useinputAlpha=False, verbosity=1):
    return _train_framework.gdNesterov(c, x0, alpha, gamma, maxEval, TOL, resetAlpha, useinputAlpha, verbosity)
kSimpleAllreduce1 = _train_framework.kSimpleAllreduce1
kSimpleAllreduce2 = _train_framework.kSimpleAllreduce2
kScatterReduce = _train_framework.kScatterReduce
kAllGather = _train_framework.kAllGather
kmaxandmin = _train_framework.kmaxandmin
decencomm1 = _train_framework.decencomm1
decencomm2 = _train_framework.decencomm2
class SumOperator(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self):
        _train_framework.SumOperator_swiginit(self, _train_framework.new_SumOperator())
    __swig_destroy__ = _train_framework.delete_SumOperator

# Register SumOperator in _train_framework:
_train_framework.SumOperator_swigregister(SumOperator)

class MinOperator(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self):
        _train_framework.MinOperator_swiginit(self, _train_framework.new_MinOperator())
    __swig_destroy__ = _train_framework.delete_MinOperator

# Register MinOperator in _train_framework:
_train_framework.MinOperator_swigregister(MinOperator)

class MaxOperator(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self):
        _train_framework.MaxOperator_swiginit(self, _train_framework.new_MaxOperator())
    __swig_destroy__ = _train_framework.delete_MaxOperator

# Register MaxOperator in _train_framework:
_train_framework.MaxOperator_swigregister(MaxOperator)

class ProductOperator(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self):
        _train_framework.ProductOperator_swiginit(self, _train_framework.new_ProductOperator())
    __swig_destroy__ = _train_framework.delete_ProductOperator

# Register ProductOperator in _train_framework:
_train_framework.ProductOperator_swigregister(ProductOperator)

class conf_util(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self):
        _train_framework.conf_util_swiginit(self, _train_framework.new_conf_util())

    def parse(self, conf_file):
        return _train_framework.conf_util_parse(self, conf_file)
    __swig_destroy__ = _train_framework.delete_conf_util

# Register conf_util in _train_framework:
_train_framework.conf_util_swigregister(conf_util)

class args_t(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, rank, size):
        _train_framework.args_t_swiginit(self, _train_framework.new_args_t(rank, size))
    myid = property(_train_framework.args_t_myid_get, _train_framework.args_t_myid_set)
    procnum = property(_train_framework.args_t_procnum_get, _train_framework.args_t_procnum_set)
    train_data_path = property(_train_framework.args_t_train_data_path_get, _train_framework.args_t_train_data_path_set)
    test_data_path = property(_train_framework.args_t_test_data_path_get, _train_framework.args_t_test_data_path_set)
    data_direction_ = property(_train_framework.args_t_data_direction__get, _train_framework.args_t_data_direction__set)
    Comm_method = property(_train_framework.args_t_Comm_method_get, _train_framework.args_t_Comm_method_set)
    maxIteration = property(_train_framework.args_t_maxIteration_get, _train_framework.args_t_maxIteration_set)
    nodesOfGroup = property(_train_framework.args_t_nodesOfGroup_get, _train_framework.args_t_nodesOfGroup_set)
    Update_method = property(_train_framework.args_t_Update_method_get, _train_framework.args_t_Update_method_set)
    Repeat_iter = property(_train_framework.args_t_Repeat_iter_get, _train_framework.args_t_Repeat_iter_set)
    rho = property(_train_framework.args_t_rho_get, _train_framework.args_t_rho_set)

    def get_args(self):
        return _train_framework.args_t_get_args(self)

    def print_args(self):
        return _train_framework.args_t_print_args(self)
    __swig_destroy__ = _train_framework.delete_args_t

# Register args_t in _train_framework:
_train_framework.args_t_swigregister(args_t)


def LeftTrim(s):
    return _train_framework.LeftTrim(s)

def RightTrim(s):
    return _train_framework.RightTrim(s)

def Trim(s):
    return _train_framework.Trim(s)
class Properties(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, path):
        _train_framework.Properties_swiginit(self, _train_framework.new_Properties(path))

    def GetString(self, property_name):
        return _train_framework.Properties_GetString(self, property_name)

    def GetInt(self, property_name):
        return _train_framework.Properties_GetInt(self, property_name)

    def GetDouble(self, property_name):
        return _train_framework.Properties_GetDouble(self, property_name)
    __swig_destroy__ = _train_framework.delete_Properties

# Register Properties in _train_framework:
_train_framework.Properties_swigregister(Properties)

class GroupStrategy(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    repeatIter = property(_train_framework.GroupStrategy_repeatIter_get, _train_framework.GroupStrategy_repeatIter_set)

    def __init__(self, iternums):
        _train_framework.GroupStrategy_swiginit(self, _train_framework.new_GroupStrategy(iternums))

    def exchangeElement(self, data, GroupNum, Group1, Group2, part):
        return _train_framework.GroupStrategy_exchangeElement(self, data, GroupNum, Group1, Group2, part)

    def divideGroup(self, nodes, groupNums):
        return _train_framework.GroupStrategy_divideGroup(self, nodes, groupNums)

    def position(self, vec, size, index):
        return _train_framework.GroupStrategy_position(self, vec, size, index)

    def findFastNodes(self, time, group, node, numsofGrup, size):
        return _train_framework.GroupStrategy_findFastNodes(self, time, group, node, numsofGrup, size)

    def changeGroup(self, data, node, fastVec, numsOfgroup, iter):
        return _train_framework.GroupStrategy_changeGroup(self, data, node, fastVec, numsOfgroup, iter)

    def MasterNodes(self, procnum, nodesOfGroup, DynamicGroup, maxIteration):
        return _train_framework.GroupStrategy_MasterNodes(self, procnum, nodesOfGroup, DynamicGroup, maxIteration)
    __swig_destroy__ = _train_framework.delete_GroupStrategy

# Register GroupStrategy in _train_framework:
_train_framework.GroupStrategy_swigregister(GroupStrategy)

class neighbors(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    neighborsNums = property(_train_framework.neighbors_neighborsNums_get, _train_framework.neighbors_neighborsNums_set)
    neighs = property(_train_framework.neighbors_neighs_get, _train_framework.neighbors_neighs_set)

    def setNeighbours(self, nums, set):
        return _train_framework.neighbors_setNeighbours(self, nums, set)

    def clearNeighbours(self):
        return _train_framework.neighbors_clearNeighbours(self)

    def __init__(self):
        _train_framework.neighbors_swiginit(self, _train_framework.new_neighbors())
    __swig_destroy__ = _train_framework.delete_neighbors

# Register neighbors in _train_framework:
_train_framework.neighbors_swigregister(neighbors)

class ADMM(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, args, train_features, ytrain, test_features, ytest, dimension, optimizer, beta, comm):
        _train_framework.ADMM_swiginit(self, _train_framework.new_ADMM(args, train_features, ytrain, test_features, ytest, dimension, optimizer, beta, comm))
    __swig_destroy__ = _train_framework.delete_ADMM

    def alpha_update(self, new_x, x_old, sumx):
        return _train_framework.ADMM_alpha_update(self, new_x, x_old, sumx)

    def group_train(self, start_time):
        return _train_framework.ADMM_group_train(self, start_time)

    def predict_comlkit(self, method):
        return _train_framework.ADMM_predict_comlkit(self, method)

    def loss_value_comlkit(self, method):
        return _train_framework.ADMM_loss_value_comlkit(self, method)

    def CreateGroup(self):
        return _train_framework.ADMM_CreateGroup(self)
    nears = property(_train_framework.ADMM_nears_get, _train_framework.ADMM_nears_set)
    sum_cal_ = property(_train_framework.ADMM_sum_cal__get, _train_framework.ADMM_sum_cal__set)
    sum_comm_ = property(_train_framework.ADMM_sum_comm__get, _train_framework.ADMM_sum_comm__set)
    quantify_part_ = property(_train_framework.ADMM_quantify_part__get, _train_framework.ADMM_quantify_part__set)
    dynamic_group_ = property(_train_framework.ADMM_dynamic_group__get, _train_framework.ADMM_dynamic_group__set)
    update_method_ = property(_train_framework.ADMM_update_method__get, _train_framework.ADMM_update_method__set)
    sparse_comm_ = property(_train_framework.ADMM_sparse_comm__get, _train_framework.ADMM_sparse_comm__set)

# Register ADMM in _train_framework:
_train_framework.ADMM_swigregister(ADMM)


def test_main(comm):
    return _train_framework.test_main(comm)

def test_main2(comm):
    return _train_framework.test_main2(comm)

def test_main3(comm):
    return _train_framework.test_main3(comm)


