�
�4�4
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
�
ArgMax

input"T
	dimension"Tidx
output"output_type"!
Ttype:
2	
"
Tidxtype0:
2	"!
output_typetype0	:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
A
BroadcastArgs
s0"T
s1"T
r0"T"
Ttype0:
2	
Z
BroadcastTo

input"T
shape"Tidx
output"T"	
Ttype"
Tidxtype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
P

ComplexAbs
x"T	
y"Tout"
Ttype0:
2"
Touttype0:
2
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

,
Cos
x"T
y"T"
Ttype:

2
�
	DecodeWav
contents	
audio
sample_rate"$
desired_channelsint���������"#
desired_samplesint���������
$
DisableCopyOnRead
resource�
R
Equal
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(�
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
A
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
?
FloorMod
x"T
y"T
z"T"
Ttype:
2	
�
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
,
Log
x"T
y"T"
Ttype:

2
�
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	"
grad_abool( "
grad_bbool( 
�
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
�
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
>
Minimum
x"T
y"T
z"T"
Ttype:
2	
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
_
Pad

input"T
paddings"	Tpaddings
output"T"	
Ttype"
	Tpaddingstype0:
2	
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
s
RFFT
input"Treal

fft_length
output"Tcomplex"
Trealtype0:
2"
Tcomplextype0:
2
f
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx" 
Tidxtype0:
2
	
&
ReadFile
filename
contents
@
ReadVariableOp
resource
value"dtype"
dtypetype�
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
A
SelectV2
	condition

t"T
e"T
output"T"	
Ttype
d
Shape

input"T&
output"out_type��out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
9
Softmax
logits"T
softmax"T"
Ttype:
2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
�
SplitV

value"T
size_splits"Tlen
	split_dim
output"T*	num_split"
	num_splitint(0"	
Ttype"
Tlentype0	:
2	
-
Sqrt
x"T
y"T"
Ttype:

2
7
Square
x"T
y"T"
Ttype:
2	
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( ""
Ttype:
2	"
Tidxtype0:
2	
-
Tanh
x"T
y"T"
Ttype:

2
�
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handle���element_dtype"
element_dtypetype"

shape_typetype:
2	
�
TensorListReserve
element_shape"
shape_type
num_elements(
handle���element_dtype"
element_dtypetype"

shape_typetype:
2	
�
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint���������
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "

debug_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �
�
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
�"serve*2.16.12v2.16.0-rc0-18-g5bc9d26649c8��
j
ConstConst*&
_output_shapes
:b2*
dtype0*%
valueBb2*  �?
l
Const_1Const*&
_output_shapes
:b2*
dtype0*%
valueBb2*����
�
batchnorm_3_/moving_varianceVarHandleOp*
_output_shapes
: *-

debug_namebatchnorm_3_/moving_variance/*
dtype0*
shape:*-
shared_namebatchnorm_3_/moving_variance
�
0batchnorm_3_/moving_variance/Read/ReadVariableOpReadVariableOpbatchnorm_3_/moving_variance*
_output_shapes
:*
dtype0
�
batchnorm_3_/moving_meanVarHandleOp*
_output_shapes
: *)

debug_namebatchnorm_3_/moving_mean/*
dtype0*
shape:*)
shared_namebatchnorm_3_/moving_mean
�
,batchnorm_3_/moving_mean/Read/ReadVariableOpReadVariableOpbatchnorm_3_/moving_mean*
_output_shapes
:*
dtype0
�
batchnorm_1_/moving_meanVarHandleOp*
_output_shapes
: *)

debug_namebatchnorm_1_/moving_mean/*
dtype0*
shape:*)
shared_namebatchnorm_1_/moving_mean
�
,batchnorm_1_/moving_mean/Read/ReadVariableOpReadVariableOpbatchnorm_1_/moving_mean*
_output_shapes
:*
dtype0
�
batchnorm_4_/moving_varianceVarHandleOp*
_output_shapes
: *-

debug_namebatchnorm_4_/moving_variance/*
dtype0*
shape: *-
shared_namebatchnorm_4_/moving_variance
�
0batchnorm_4_/moving_variance/Read/ReadVariableOpReadVariableOpbatchnorm_4_/moving_variance*
_output_shapes
: *
dtype0
�
batchnorm_5_/moving_meanVarHandleOp*
_output_shapes
: *)

debug_namebatchnorm_5_/moving_mean/*
dtype0*
shape:P*)
shared_namebatchnorm_5_/moving_mean
�
,batchnorm_5_/moving_mean/Read/ReadVariableOpReadVariableOpbatchnorm_5_/moving_mean*
_output_shapes
:P*
dtype0
�
batchnorm_4_/moving_meanVarHandleOp*
_output_shapes
: *)

debug_namebatchnorm_4_/moving_mean/*
dtype0*
shape: *)
shared_namebatchnorm_4_/moving_mean
�
,batchnorm_4_/moving_mean/Read/ReadVariableOpReadVariableOpbatchnorm_4_/moving_mean*
_output_shapes
: *
dtype0
�
batchnorm_2_/moving_meanVarHandleOp*
_output_shapes
: *)

debug_namebatchnorm_2_/moving_mean/*
dtype0*
shape:*)
shared_namebatchnorm_2_/moving_mean
�
,batchnorm_2_/moving_mean/Read/ReadVariableOpReadVariableOpbatchnorm_2_/moving_mean*
_output_shapes
:*
dtype0
�
batchnorm_1_/moving_varianceVarHandleOp*
_output_shapes
: *-

debug_namebatchnorm_1_/moving_variance/*
dtype0*
shape:*-
shared_namebatchnorm_1_/moving_variance
�
0batchnorm_1_/moving_variance/Read/ReadVariableOpReadVariableOpbatchnorm_1_/moving_variance*
_output_shapes
:*
dtype0
�
batchnorm_2_/moving_varianceVarHandleOp*
_output_shapes
: *-

debug_namebatchnorm_2_/moving_variance/*
dtype0*
shape:*-
shared_namebatchnorm_2_/moving_variance
�
0batchnorm_2_/moving_variance/Read/ReadVariableOpReadVariableOpbatchnorm_2_/moving_variance*
_output_shapes
:*
dtype0
�
batchnorm_5_/moving_varianceVarHandleOp*
_output_shapes
: *-

debug_namebatchnorm_5_/moving_variance/*
dtype0*
shape:P*-
shared_namebatchnorm_5_/moving_variance
�
0batchnorm_5_/moving_variance/Read/ReadVariableOpReadVariableOpbatchnorm_5_/moving_variance*
_output_shapes
:P*
dtype0
�
conv_4_/biasVarHandleOp*
_output_shapes
: *

debug_nameconv_4_/bias/*
dtype0*
shape: *
shared_nameconv_4_/bias
i
 conv_4_/bias/Read/ReadVariableOpReadVariableOpconv_4_/bias*
_output_shapes
: *
dtype0
�
conv_2_/biasVarHandleOp*
_output_shapes
: *

debug_nameconv_2_/bias/*
dtype0*
shape:*
shared_nameconv_2_/bias
i
 conv_2_/bias/Read/ReadVariableOpReadVariableOpconv_2_/bias*
_output_shapes
:*
dtype0
�
batchnorm_1_/gammaVarHandleOp*
_output_shapes
: *#

debug_namebatchnorm_1_/gamma/*
dtype0*
shape:*#
shared_namebatchnorm_1_/gamma
u
&batchnorm_1_/gamma/Read/ReadVariableOpReadVariableOpbatchnorm_1_/gamma*
_output_shapes
:*
dtype0
�
batchnorm_5_/betaVarHandleOp*
_output_shapes
: *"

debug_namebatchnorm_5_/beta/*
dtype0*
shape:P*"
shared_namebatchnorm_5_/beta
s
%batchnorm_5_/beta/Read/ReadVariableOpReadVariableOpbatchnorm_5_/beta*
_output_shapes
:P*
dtype0
�
batchnorm_4_/betaVarHandleOp*
_output_shapes
: *"

debug_namebatchnorm_4_/beta/*
dtype0*
shape: *"
shared_namebatchnorm_4_/beta
s
%batchnorm_4_/beta/Read/ReadVariableOpReadVariableOpbatchnorm_4_/beta*
_output_shapes
: *
dtype0
�
batchnorm_2_/betaVarHandleOp*
_output_shapes
: *"

debug_namebatchnorm_2_/beta/*
dtype0*
shape:*"
shared_namebatchnorm_2_/beta
s
%batchnorm_2_/beta/Read/ReadVariableOpReadVariableOpbatchnorm_2_/beta*
_output_shapes
:*
dtype0
�

fc_2_/biasVarHandleOp*
_output_shapes
: *

debug_namefc_2_/bias/*
dtype0*
shape:*
shared_name
fc_2_/bias
e
fc_2_/bias/Read/ReadVariableOpReadVariableOp
fc_2_/bias*
_output_shapes
:*
dtype0
�
batchnorm_3_/gammaVarHandleOp*
_output_shapes
: *#

debug_namebatchnorm_3_/gamma/*
dtype0*
shape:*#
shared_namebatchnorm_3_/gamma
u
&batchnorm_3_/gamma/Read/ReadVariableOpReadVariableOpbatchnorm_3_/gamma*
_output_shapes
:*
dtype0
�

fc_1_/biasVarHandleOp*
_output_shapes
: *

debug_namefc_1_/bias/*
dtype0*
shape:*
shared_name
fc_1_/bias
e
fc_1_/bias/Read/ReadVariableOpReadVariableOp
fc_1_/bias*
_output_shapes
:*
dtype0
�
lstm_/lstm_cell/biasVarHandleOp*
_output_shapes
: *%

debug_namelstm_/lstm_cell/bias/*
dtype0*
shape:�*%
shared_namelstm_/lstm_cell/bias
z
(lstm_/lstm_cell/bias/Read/ReadVariableOpReadVariableOplstm_/lstm_cell/bias*
_output_shapes	
:�*
dtype0
�
fc_2_/kernelVarHandleOp*
_output_shapes
: *

debug_namefc_2_/kernel/*
dtype0*
shape
:*
shared_namefc_2_/kernel
m
 fc_2_/kernel/Read/ReadVariableOpReadVariableOpfc_2_/kernel*
_output_shapes

:*
dtype0
�
 lstm_/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *1

debug_name#!lstm_/lstm_cell/recurrent_kernel/*
dtype0*
shape:
��*1
shared_name" lstm_/lstm_cell/recurrent_kernel
�
4lstm_/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp lstm_/lstm_cell/recurrent_kernel* 
_output_shapes
:
��*
dtype0
�
batchnorm_5_/gammaVarHandleOp*
_output_shapes
: *#

debug_namebatchnorm_5_/gamma/*
dtype0*
shape:P*#
shared_namebatchnorm_5_/gamma
u
&batchnorm_5_/gamma/Read/ReadVariableOpReadVariableOpbatchnorm_5_/gamma*
_output_shapes
:P*
dtype0
�
batchnorm_4_/gammaVarHandleOp*
_output_shapes
: *#

debug_namebatchnorm_4_/gamma/*
dtype0*
shape: *#
shared_namebatchnorm_4_/gamma
u
&batchnorm_4_/gamma/Read/ReadVariableOpReadVariableOpbatchnorm_4_/gamma*
_output_shapes
: *
dtype0
�
batchnorm_2_/gammaVarHandleOp*
_output_shapes
: *#

debug_namebatchnorm_2_/gamma/*
dtype0*
shape:*#
shared_namebatchnorm_2_/gamma
u
&batchnorm_2_/gamma/Read/ReadVariableOpReadVariableOpbatchnorm_2_/gamma*
_output_shapes
:*
dtype0
�
conv_1_/biasVarHandleOp*
_output_shapes
: *

debug_nameconv_1_/bias/*
dtype0*
shape:*
shared_nameconv_1_/bias
i
 conv_1_/bias/Read/ReadVariableOpReadVariableOpconv_1_/bias*
_output_shapes
:*
dtype0
�
fc_1_/kernelVarHandleOp*
_output_shapes
: *

debug_namefc_1_/kernel/*
dtype0*
shape:	�*
shared_namefc_1_/kernel
n
 fc_1_/kernel/Read/ReadVariableOpReadVariableOpfc_1_/kernel*
_output_shapes
:	�*
dtype0
�
lstm_/lstm_cell/kernelVarHandleOp*
_output_shapes
: *'

debug_namelstm_/lstm_cell/kernel/*
dtype0*
shape:
��*'
shared_namelstm_/lstm_cell/kernel
�
*lstm_/lstm_cell/kernel/Read/ReadVariableOpReadVariableOplstm_/lstm_cell/kernel* 
_output_shapes
:
��*
dtype0
�
conv_3_/kernelVarHandleOp*
_output_shapes
: *

debug_nameconv_3_/kernel/*
dtype0*
shape:*
shared_nameconv_3_/kernel
y
"conv_3_/kernel/Read/ReadVariableOpReadVariableOpconv_3_/kernel*&
_output_shapes
:*
dtype0
�
conv_5_/kernelVarHandleOp*
_output_shapes
: *

debug_nameconv_5_/kernel/*
dtype0*
shape: P*
shared_nameconv_5_/kernel
y
"conv_5_/kernel/Read/ReadVariableOpReadVariableOpconv_5_/kernel*&
_output_shapes
: P*
dtype0
�
conv_3_/biasVarHandleOp*
_output_shapes
: *

debug_nameconv_3_/bias/*
dtype0*
shape:*
shared_nameconv_3_/bias
i
 conv_3_/bias/Read/ReadVariableOpReadVariableOpconv_3_/bias*
_output_shapes
:*
dtype0
�
conv_1_/kernelVarHandleOp*
_output_shapes
: *

debug_nameconv_1_/kernel/*
dtype0*
shape:*
shared_nameconv_1_/kernel
y
"conv_1_/kernel/Read/ReadVariableOpReadVariableOpconv_1_/kernel*&
_output_shapes
:*
dtype0
�
conv_5_/biasVarHandleOp*
_output_shapes
: *

debug_nameconv_5_/bias/*
dtype0*
shape:P*
shared_nameconv_5_/bias
i
 conv_5_/bias/Read/ReadVariableOpReadVariableOpconv_5_/bias*
_output_shapes
:P*
dtype0
�
conv_4_/kernelVarHandleOp*
_output_shapes
: *

debug_nameconv_4_/kernel/*
dtype0*
shape: *
shared_nameconv_4_/kernel
y
"conv_4_/kernel/Read/ReadVariableOpReadVariableOpconv_4_/kernel*&
_output_shapes
: *
dtype0
�
batchnorm_3_/betaVarHandleOp*
_output_shapes
: *"

debug_namebatchnorm_3_/beta/*
dtype0*
shape:*"
shared_namebatchnorm_3_/beta
s
%batchnorm_3_/beta/Read/ReadVariableOpReadVariableOpbatchnorm_3_/beta*
_output_shapes
:*
dtype0
�
conv_2_/kernelVarHandleOp*
_output_shapes
: *

debug_nameconv_2_/kernel/*
dtype0*
shape:*
shared_nameconv_2_/kernel
y
"conv_2_/kernel/Read/ReadVariableOpReadVariableOpconv_2_/kernel*&
_output_shapes
:*
dtype0
�
batchnorm_1_/betaVarHandleOp*
_output_shapes
: *"

debug_namebatchnorm_1_/beta/*
dtype0*
shape:*"
shared_namebatchnorm_1_/beta
s
%batchnorm_1_/beta/Read/ReadVariableOpReadVariableOpbatchnorm_1_/beta*
_output_shapes
:*
dtype0
�
fc_2_/bias_1VarHandleOp*
_output_shapes
: *

debug_namefc_2_/bias_1/*
dtype0*
shape:*
shared_namefc_2_/bias_1
i
 fc_2_/bias_1/Read/ReadVariableOpReadVariableOpfc_2_/bias_1*
_output_shapes
:*
dtype0
�
fc_2_/kernel_1VarHandleOp*
_output_shapes
: *

debug_namefc_2_/kernel_1/*
dtype0*
shape
:*
shared_namefc_2_/kernel_1
q
"fc_2_/kernel_1/Read/ReadVariableOpReadVariableOpfc_2_/kernel_1*
_output_shapes

:*
dtype0
�
seed_generator_stateVarHandleOp*
_output_shapes
: *%

debug_nameseed_generator_state/*
dtype0*
shape:*%
shared_nameseed_generator_state
y
(seed_generator_state/Read/ReadVariableOpReadVariableOpseed_generator_state*
_output_shapes
:*
dtype0
�
fc_1_/bias_1VarHandleOp*
_output_shapes
: *

debug_namefc_1_/bias_1/*
dtype0*
shape:*
shared_namefc_1_/bias_1
i
 fc_1_/bias_1/Read/ReadVariableOpReadVariableOpfc_1_/bias_1*
_output_shapes
:*
dtype0
�
fc_1_/kernel_1VarHandleOp*
_output_shapes
: *

debug_namefc_1_/kernel_1/*
dtype0*
shape:	�*
shared_namefc_1_/kernel_1
r
"fc_1_/kernel_1/Read/ReadVariableOpReadVariableOpfc_1_/kernel_1*
_output_shapes
:	�*
dtype0
�
seed_generator_state_1VarHandleOp*
_output_shapes
: *'

debug_nameseed_generator_state_1/*
dtype0*
shape:*'
shared_nameseed_generator_state_1
}
*seed_generator_state_1/Read/ReadVariableOpReadVariableOpseed_generator_state_1*
_output_shapes
:*
dtype0
�
lstm_/lstm_cell/bias_1VarHandleOp*
_output_shapes
: *'

debug_namelstm_/lstm_cell/bias_1/*
dtype0*
shape:�*'
shared_namelstm_/lstm_cell/bias_1
~
*lstm_/lstm_cell/bias_1/Read/ReadVariableOpReadVariableOplstm_/lstm_cell/bias_1*
_output_shapes	
:�*
dtype0
�
"lstm_/lstm_cell/recurrent_kernel_1VarHandleOp*
_output_shapes
: *3

debug_name%#lstm_/lstm_cell/recurrent_kernel_1/*
dtype0*
shape:
��*3
shared_name$"lstm_/lstm_cell/recurrent_kernel_1
�
6lstm_/lstm_cell/recurrent_kernel_1/Read/ReadVariableOpReadVariableOp"lstm_/lstm_cell/recurrent_kernel_1* 
_output_shapes
:
��*
dtype0
�
lstm_/lstm_cell/kernel_1VarHandleOp*
_output_shapes
: *)

debug_namelstm_/lstm_cell/kernel_1/*
dtype0*
shape:
��*)
shared_namelstm_/lstm_cell/kernel_1
�
,lstm_/lstm_cell/kernel_1/Read/ReadVariableOpReadVariableOplstm_/lstm_cell/kernel_1* 
_output_shapes
:
��*
dtype0
�
batchnorm_5_/moving_variance_1VarHandleOp*
_output_shapes
: */

debug_name!batchnorm_5_/moving_variance_1/*
dtype0*
shape:P*/
shared_name batchnorm_5_/moving_variance_1
�
2batchnorm_5_/moving_variance_1/Read/ReadVariableOpReadVariableOpbatchnorm_5_/moving_variance_1*
_output_shapes
:P*
dtype0
�
batchnorm_5_/moving_mean_1VarHandleOp*
_output_shapes
: *+

debug_namebatchnorm_5_/moving_mean_1/*
dtype0*
shape:P*+
shared_namebatchnorm_5_/moving_mean_1
�
.batchnorm_5_/moving_mean_1/Read/ReadVariableOpReadVariableOpbatchnorm_5_/moving_mean_1*
_output_shapes
:P*
dtype0
�
batchnorm_5_/beta_1VarHandleOp*
_output_shapes
: *$

debug_namebatchnorm_5_/beta_1/*
dtype0*
shape:P*$
shared_namebatchnorm_5_/beta_1
w
'batchnorm_5_/beta_1/Read/ReadVariableOpReadVariableOpbatchnorm_5_/beta_1*
_output_shapes
:P*
dtype0
�
batchnorm_5_/gamma_1VarHandleOp*
_output_shapes
: *%

debug_namebatchnorm_5_/gamma_1/*
dtype0*
shape:P*%
shared_namebatchnorm_5_/gamma_1
y
(batchnorm_5_/gamma_1/Read/ReadVariableOpReadVariableOpbatchnorm_5_/gamma_1*
_output_shapes
:P*
dtype0
�
conv_5_/bias_1VarHandleOp*
_output_shapes
: *

debug_nameconv_5_/bias_1/*
dtype0*
shape:P*
shared_nameconv_5_/bias_1
m
"conv_5_/bias_1/Read/ReadVariableOpReadVariableOpconv_5_/bias_1*
_output_shapes
:P*
dtype0
�
conv_5_/kernel_1VarHandleOp*
_output_shapes
: *!

debug_nameconv_5_/kernel_1/*
dtype0*
shape: P*!
shared_nameconv_5_/kernel_1
}
$conv_5_/kernel_1/Read/ReadVariableOpReadVariableOpconv_5_/kernel_1*&
_output_shapes
: P*
dtype0
�
batchnorm_4_/moving_variance_1VarHandleOp*
_output_shapes
: */

debug_name!batchnorm_4_/moving_variance_1/*
dtype0*
shape: */
shared_name batchnorm_4_/moving_variance_1
�
2batchnorm_4_/moving_variance_1/Read/ReadVariableOpReadVariableOpbatchnorm_4_/moving_variance_1*
_output_shapes
: *
dtype0
�
batchnorm_4_/moving_mean_1VarHandleOp*
_output_shapes
: *+

debug_namebatchnorm_4_/moving_mean_1/*
dtype0*
shape: *+
shared_namebatchnorm_4_/moving_mean_1
�
.batchnorm_4_/moving_mean_1/Read/ReadVariableOpReadVariableOpbatchnorm_4_/moving_mean_1*
_output_shapes
: *
dtype0
�
batchnorm_4_/beta_1VarHandleOp*
_output_shapes
: *$

debug_namebatchnorm_4_/beta_1/*
dtype0*
shape: *$
shared_namebatchnorm_4_/beta_1
w
'batchnorm_4_/beta_1/Read/ReadVariableOpReadVariableOpbatchnorm_4_/beta_1*
_output_shapes
: *
dtype0
�
batchnorm_4_/gamma_1VarHandleOp*
_output_shapes
: *%

debug_namebatchnorm_4_/gamma_1/*
dtype0*
shape: *%
shared_namebatchnorm_4_/gamma_1
y
(batchnorm_4_/gamma_1/Read/ReadVariableOpReadVariableOpbatchnorm_4_/gamma_1*
_output_shapes
: *
dtype0
�
conv_4_/bias_1VarHandleOp*
_output_shapes
: *

debug_nameconv_4_/bias_1/*
dtype0*
shape: *
shared_nameconv_4_/bias_1
m
"conv_4_/bias_1/Read/ReadVariableOpReadVariableOpconv_4_/bias_1*
_output_shapes
: *
dtype0
�
conv_4_/kernel_1VarHandleOp*
_output_shapes
: *!

debug_nameconv_4_/kernel_1/*
dtype0*
shape: *!
shared_nameconv_4_/kernel_1
}
$conv_4_/kernel_1/Read/ReadVariableOpReadVariableOpconv_4_/kernel_1*&
_output_shapes
: *
dtype0
�
batchnorm_3_/moving_variance_1VarHandleOp*
_output_shapes
: */

debug_name!batchnorm_3_/moving_variance_1/*
dtype0*
shape:*/
shared_name batchnorm_3_/moving_variance_1
�
2batchnorm_3_/moving_variance_1/Read/ReadVariableOpReadVariableOpbatchnorm_3_/moving_variance_1*
_output_shapes
:*
dtype0
�
batchnorm_3_/moving_mean_1VarHandleOp*
_output_shapes
: *+

debug_namebatchnorm_3_/moving_mean_1/*
dtype0*
shape:*+
shared_namebatchnorm_3_/moving_mean_1
�
.batchnorm_3_/moving_mean_1/Read/ReadVariableOpReadVariableOpbatchnorm_3_/moving_mean_1*
_output_shapes
:*
dtype0
�
batchnorm_3_/beta_1VarHandleOp*
_output_shapes
: *$

debug_namebatchnorm_3_/beta_1/*
dtype0*
shape:*$
shared_namebatchnorm_3_/beta_1
w
'batchnorm_3_/beta_1/Read/ReadVariableOpReadVariableOpbatchnorm_3_/beta_1*
_output_shapes
:*
dtype0
�
batchnorm_3_/gamma_1VarHandleOp*
_output_shapes
: *%

debug_namebatchnorm_3_/gamma_1/*
dtype0*
shape:*%
shared_namebatchnorm_3_/gamma_1
y
(batchnorm_3_/gamma_1/Read/ReadVariableOpReadVariableOpbatchnorm_3_/gamma_1*
_output_shapes
:*
dtype0
�
conv_3_/bias_1VarHandleOp*
_output_shapes
: *

debug_nameconv_3_/bias_1/*
dtype0*
shape:*
shared_nameconv_3_/bias_1
m
"conv_3_/bias_1/Read/ReadVariableOpReadVariableOpconv_3_/bias_1*
_output_shapes
:*
dtype0
�
conv_3_/kernel_1VarHandleOp*
_output_shapes
: *!

debug_nameconv_3_/kernel_1/*
dtype0*
shape:*!
shared_nameconv_3_/kernel_1
}
$conv_3_/kernel_1/Read/ReadVariableOpReadVariableOpconv_3_/kernel_1*&
_output_shapes
:*
dtype0
�
batchnorm_2_/moving_variance_1VarHandleOp*
_output_shapes
: */

debug_name!batchnorm_2_/moving_variance_1/*
dtype0*
shape:*/
shared_name batchnorm_2_/moving_variance_1
�
2batchnorm_2_/moving_variance_1/Read/ReadVariableOpReadVariableOpbatchnorm_2_/moving_variance_1*
_output_shapes
:*
dtype0
�
batchnorm_2_/moving_mean_1VarHandleOp*
_output_shapes
: *+

debug_namebatchnorm_2_/moving_mean_1/*
dtype0*
shape:*+
shared_namebatchnorm_2_/moving_mean_1
�
.batchnorm_2_/moving_mean_1/Read/ReadVariableOpReadVariableOpbatchnorm_2_/moving_mean_1*
_output_shapes
:*
dtype0
�
batchnorm_2_/beta_1VarHandleOp*
_output_shapes
: *$

debug_namebatchnorm_2_/beta_1/*
dtype0*
shape:*$
shared_namebatchnorm_2_/beta_1
w
'batchnorm_2_/beta_1/Read/ReadVariableOpReadVariableOpbatchnorm_2_/beta_1*
_output_shapes
:*
dtype0
�
batchnorm_2_/gamma_1VarHandleOp*
_output_shapes
: *%

debug_namebatchnorm_2_/gamma_1/*
dtype0*
shape:*%
shared_namebatchnorm_2_/gamma_1
y
(batchnorm_2_/gamma_1/Read/ReadVariableOpReadVariableOpbatchnorm_2_/gamma_1*
_output_shapes
:*
dtype0
�
conv_2_/bias_1VarHandleOp*
_output_shapes
: *

debug_nameconv_2_/bias_1/*
dtype0*
shape:*
shared_nameconv_2_/bias_1
m
"conv_2_/bias_1/Read/ReadVariableOpReadVariableOpconv_2_/bias_1*
_output_shapes
:*
dtype0
�
conv_2_/kernel_1VarHandleOp*
_output_shapes
: *!

debug_nameconv_2_/kernel_1/*
dtype0*
shape:*!
shared_nameconv_2_/kernel_1
}
$conv_2_/kernel_1/Read/ReadVariableOpReadVariableOpconv_2_/kernel_1*&
_output_shapes
:*
dtype0
�
batchnorm_1_/moving_variance_1VarHandleOp*
_output_shapes
: */

debug_name!batchnorm_1_/moving_variance_1/*
dtype0*
shape:*/
shared_name batchnorm_1_/moving_variance_1
�
2batchnorm_1_/moving_variance_1/Read/ReadVariableOpReadVariableOpbatchnorm_1_/moving_variance_1*
_output_shapes
:*
dtype0
�
batchnorm_1_/moving_mean_1VarHandleOp*
_output_shapes
: *+

debug_namebatchnorm_1_/moving_mean_1/*
dtype0*
shape:*+
shared_namebatchnorm_1_/moving_mean_1
�
.batchnorm_1_/moving_mean_1/Read/ReadVariableOpReadVariableOpbatchnorm_1_/moving_mean_1*
_output_shapes
:*
dtype0
�
batchnorm_1_/beta_1VarHandleOp*
_output_shapes
: *$

debug_namebatchnorm_1_/beta_1/*
dtype0*
shape:*$
shared_namebatchnorm_1_/beta_1
w
'batchnorm_1_/beta_1/Read/ReadVariableOpReadVariableOpbatchnorm_1_/beta_1*
_output_shapes
:*
dtype0
�
batchnorm_1_/gamma_1VarHandleOp*
_output_shapes
: *%

debug_namebatchnorm_1_/gamma_1/*
dtype0*
shape:*%
shared_namebatchnorm_1_/gamma_1
y
(batchnorm_1_/gamma_1/Read/ReadVariableOpReadVariableOpbatchnorm_1_/gamma_1*
_output_shapes
:*
dtype0
�
conv_1_/bias_1VarHandleOp*
_output_shapes
: *

debug_nameconv_1_/bias_1/*
dtype0*
shape:*
shared_nameconv_1_/bias_1
m
"conv_1_/bias_1/Read/ReadVariableOpReadVariableOpconv_1_/bias_1*
_output_shapes
:*
dtype0
�
conv_1_/kernel_1VarHandleOp*
_output_shapes
: *!

debug_nameconv_1_/kernel_1/*
dtype0*
shape:*!
shared_nameconv_1_/kernel_1
}
$conv_1_/kernel_1/Read/ReadVariableOpReadVariableOpconv_1_/kernel_1*&
_output_shapes
:*
dtype0
�
imageinput_/countVarHandleOp*
_output_shapes
: *"

debug_nameimageinput_/count/*
dtype0	*
shape: *"
shared_nameimageinput_/count
o
%imageinput_/count/Read/ReadVariableOpReadVariableOpimageinput_/count*
_output_shapes
: *
dtype0	
�
imageinput_/varianceVarHandleOp*
_output_shapes
: *%

debug_nameimageinput_/variance/*
dtype0*
shape:b2*%
shared_nameimageinput_/variance
�
(imageinput_/variance/Read/ReadVariableOpReadVariableOpimageinput_/variance*"
_output_shapes
:b2*
dtype0
�
imageinput_/meanVarHandleOp*
_output_shapes
: *!

debug_nameimageinput_/mean/*
dtype0*
shape:b2*!
shared_nameimageinput_/mean
y
$imageinput_/mean/Read/ReadVariableOpReadVariableOpimageinput_/mean*"
_output_shapes
:b2*
dtype0
V
serving_default_inputPlaceholder*
_output_shapes
: *
dtype0*
shape: 
�	
StatefulPartitionedCallStatefulPartitionedCallserving_default_inputConst_1Constconv_1_/kernel_1conv_1_/bias_1batchnorm_1_/moving_mean_1batchnorm_1_/moving_variance_1batchnorm_1_/gamma_1batchnorm_1_/beta_1conv_2_/kernel_1conv_2_/bias_1batchnorm_2_/moving_mean_1batchnorm_2_/moving_variance_1batchnorm_2_/gamma_1batchnorm_2_/beta_1conv_3_/kernel_1conv_3_/bias_1batchnorm_3_/moving_mean_1batchnorm_3_/moving_variance_1batchnorm_3_/gamma_1batchnorm_3_/beta_1conv_4_/kernel_1conv_4_/bias_1batchnorm_4_/moving_mean_1batchnorm_4_/moving_variance_1batchnorm_4_/gamma_1batchnorm_4_/beta_1conv_5_/kernel_1conv_5_/bias_1batchnorm_5_/moving_mean_1batchnorm_5_/moving_variance_1batchnorm_5_/gamma_1batchnorm_5_/beta_1lstm_/lstm_cell/kernel_1"lstm_/lstm_cell/recurrent_kernel_1lstm_/lstm_cell/bias_1fc_1_/kernel_1fc_1_/bias_1fc_2_/kernel_1fc_2_/bias_1*3
Tin,
*2(*
Tout
2	*
_collective_manager_ids
 *E
_output_shapes3
1:���������:���������:���������*G
_read_only_resource_inputs)
'%	
 !"#$%&'*-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference_signature_wrapper_2485
x
serving_float_inputPlaceholder*(
_output_shapes
:����������}*
dtype0*
shape:����������}
�	
StatefulPartitionedCall_1StatefulPartitionedCallserving_float_inputConst_1Constconv_1_/kernel_1conv_1_/bias_1batchnorm_1_/moving_mean_1batchnorm_1_/moving_variance_1batchnorm_1_/gamma_1batchnorm_1_/beta_1conv_2_/kernel_1conv_2_/bias_1batchnorm_2_/moving_mean_1batchnorm_2_/moving_variance_1batchnorm_2_/gamma_1batchnorm_2_/beta_1conv_3_/kernel_1conv_3_/bias_1batchnorm_3_/moving_mean_1batchnorm_3_/moving_variance_1batchnorm_3_/gamma_1batchnorm_3_/beta_1conv_4_/kernel_1conv_4_/bias_1batchnorm_4_/moving_mean_1batchnorm_4_/moving_variance_1batchnorm_4_/gamma_1batchnorm_4_/beta_1conv_5_/kernel_1conv_5_/bias_1batchnorm_5_/moving_mean_1batchnorm_5_/moving_variance_1batchnorm_5_/gamma_1batchnorm_5_/beta_1lstm_/lstm_cell/kernel_1"lstm_/lstm_cell/recurrent_kernel_1lstm_/lstm_cell/bias_1fc_1_/kernel_1fc_1_/bias_1fc_2_/kernel_1fc_2_/bias_1*3
Tin,
*2(*
Tout
2	*
_collective_manager_ids
 *E
_output_shapes3
1:���������:���������:���������*G
_read_only_resource_inputs)
'%	
 !"#$%&'*-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference_signature_wrapper_2573

NoOpNoOp
�H
Const_2Const"/device:CPU:0*
_output_shapes
: *
dtype0*�G
value�GB�G B�G
Z
	model
__call__
call_float_input
call_string_input

signatures*
�
	variables
trainable_variables
non_trainable_variables
	_all_variables

_misc_assets

signatures
#_self_saveable_object_factories
	serve*

trace_0
trace_1* 
 
	capture_0
	capture_1* 
 
	capture_0
	capture_1* 
*
serving_default
serving_float* 
�
0
1
2
3
4
5
6
7
8
9
10
11
 12
!13
"14
#15
$16
%17
&18
'19
(20
)21
*22
+23
,24
-25
.26
/27
028
129
230
331
432
533
634
735
836
937
:38
;39
<40
=41*
�
0
1
2
3
4
5
6
 7
#8
$9
%10
&11
)12
*13
+14
,15
/16
017
118
219
520
621
722
923
:24
<25
=26*
r
0
1
2
3
4
!5
"6
'7
(8
-9
.10
311
412
813
;14*
�
>0
?1
@2
A3
B4
C5
D6
E7
F8
G9
H10
I11
J12
K13
L14
M15
N16
O17
P18
Q19
R20
S21
T22
U23
V24
W25
X26
Y27
Z28
[29
\30
]31
^32
_33
`34
a35
b36*
* 
"
	cserve
dserving_default* 
* 

etrace_0* 
 
	capture_0
	capture_1* 
 
	capture_0
	capture_1* 
* 
* 
 
	capture_0
	capture_1* 
 
	capture_0
	capture_1* 
VP
VARIABLE_VALUEimageinput_/mean,model/variables/0/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEimageinput_/variance,model/variables/1/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEimageinput_/count,model/variables/2/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEconv_1_/kernel_1,model/variables/3/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEconv_1_/bias_1,model/variables/4/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEbatchnorm_1_/gamma_1,model/variables/5/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEbatchnorm_1_/beta_1,model/variables/6/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEbatchnorm_1_/moving_mean_1,model/variables/7/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEbatchnorm_1_/moving_variance_1,model/variables/8/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEconv_2_/kernel_1,model/variables/9/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEconv_2_/bias_1-model/variables/10/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEbatchnorm_2_/gamma_1-model/variables/11/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEbatchnorm_2_/beta_1-model/variables/12/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEbatchnorm_2_/moving_mean_1-model/variables/13/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEbatchnorm_2_/moving_variance_1-model/variables/14/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEconv_3_/kernel_1-model/variables/15/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEconv_3_/bias_1-model/variables/16/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEbatchnorm_3_/gamma_1-model/variables/17/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEbatchnorm_3_/beta_1-model/variables/18/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEbatchnorm_3_/moving_mean_1-model/variables/19/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEbatchnorm_3_/moving_variance_1-model/variables/20/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEconv_4_/kernel_1-model/variables/21/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEconv_4_/bias_1-model/variables/22/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEbatchnorm_4_/gamma_1-model/variables/23/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEbatchnorm_4_/beta_1-model/variables/24/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEbatchnorm_4_/moving_mean_1-model/variables/25/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEbatchnorm_4_/moving_variance_1-model/variables/26/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEconv_5_/kernel_1-model/variables/27/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEconv_5_/bias_1-model/variables/28/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEbatchnorm_5_/gamma_1-model/variables/29/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEbatchnorm_5_/beta_1-model/variables/30/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEbatchnorm_5_/moving_mean_1-model/variables/31/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEbatchnorm_5_/moving_variance_1-model/variables/32/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUElstm_/lstm_cell/kernel_1-model/variables/33/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUE"lstm_/lstm_cell/recurrent_kernel_1-model/variables/34/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUElstm_/lstm_cell/bias_1-model/variables/35/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEseed_generator_state_1-model/variables/36/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEfc_1_/kernel_1-model/variables/37/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEfc_1_/bias_1-model/variables/38/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEseed_generator_state-model/variables/39/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEfc_2_/kernel_1-model/variables/40/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEfc_2_/bias_1-model/variables/41/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEbatchnorm_1_/beta1model/_all_variables/0/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEconv_2_/kernel1model/_all_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEbatchnorm_3_/beta1model/_all_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEconv_4_/kernel1model/_all_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEconv_5_/bias1model/_all_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEconv_1_/kernel1model/_all_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEconv_3_/bias1model/_all_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEconv_5_/kernel1model/_all_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEconv_3_/kernel1model/_all_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUElstm_/lstm_cell/kernel1model/_all_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEfc_1_/kernel2model/_all_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEconv_1_/bias2model/_all_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEbatchnorm_2_/gamma2model/_all_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEbatchnorm_4_/gamma2model/_all_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEbatchnorm_5_/gamma2model/_all_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE lstm_/lstm_cell/recurrent_kernel2model/_all_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEfc_2_/kernel2model/_all_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUElstm_/lstm_cell/bias2model/_all_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUE
fc_1_/bias2model/_all_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEbatchnorm_3_/gamma2model/_all_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUE
fc_2_/bias2model/_all_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEbatchnorm_2_/beta2model/_all_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEbatchnorm_4_/beta2model/_all_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEbatchnorm_5_/beta2model/_all_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEbatchnorm_1_/gamma2model/_all_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEconv_2_/bias2model/_all_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEconv_4_/bias2model/_all_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatchnorm_5_/moving_variance2model/_all_variables/27/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatchnorm_2_/moving_variance2model/_all_variables/28/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatchnorm_1_/moving_variance2model/_all_variables/29/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEbatchnorm_2_/moving_mean2model/_all_variables/30/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEbatchnorm_4_/moving_mean2model/_all_variables/31/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEbatchnorm_5_/moving_mean2model/_all_variables/32/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatchnorm_4_/moving_variance2model/_all_variables/33/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEbatchnorm_1_/moving_mean2model/_all_variables/34/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEbatchnorm_3_/moving_mean2model/_all_variables/35/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatchnorm_3_/moving_variance2model/_all_variables/36/.ATTRIBUTES/VARIABLE_VALUE*
 
	capture_0
	capture_1* 
 
	capture_0
	capture_1* 
 
	capture_0
	capture_1* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameimageinput_/meanimageinput_/varianceimageinput_/countconv_1_/kernel_1conv_1_/bias_1batchnorm_1_/gamma_1batchnorm_1_/beta_1batchnorm_1_/moving_mean_1batchnorm_1_/moving_variance_1conv_2_/kernel_1conv_2_/bias_1batchnorm_2_/gamma_1batchnorm_2_/beta_1batchnorm_2_/moving_mean_1batchnorm_2_/moving_variance_1conv_3_/kernel_1conv_3_/bias_1batchnorm_3_/gamma_1batchnorm_3_/beta_1batchnorm_3_/moving_mean_1batchnorm_3_/moving_variance_1conv_4_/kernel_1conv_4_/bias_1batchnorm_4_/gamma_1batchnorm_4_/beta_1batchnorm_4_/moving_mean_1batchnorm_4_/moving_variance_1conv_5_/kernel_1conv_5_/bias_1batchnorm_5_/gamma_1batchnorm_5_/beta_1batchnorm_5_/moving_mean_1batchnorm_5_/moving_variance_1lstm_/lstm_cell/kernel_1"lstm_/lstm_cell/recurrent_kernel_1lstm_/lstm_cell/bias_1seed_generator_state_1fc_1_/kernel_1fc_1_/bias_1seed_generator_statefc_2_/kernel_1fc_2_/bias_1batchnorm_1_/betaconv_2_/kernelbatchnorm_3_/betaconv_4_/kernelconv_5_/biasconv_1_/kernelconv_3_/biasconv_5_/kernelconv_3_/kernellstm_/lstm_cell/kernelfc_1_/kernelconv_1_/biasbatchnorm_2_/gammabatchnorm_4_/gammabatchnorm_5_/gamma lstm_/lstm_cell/recurrent_kernelfc_2_/kernellstm_/lstm_cell/bias
fc_1_/biasbatchnorm_3_/gamma
fc_2_/biasbatchnorm_2_/betabatchnorm_4_/betabatchnorm_5_/betabatchnorm_1_/gammaconv_2_/biasconv_4_/biasbatchnorm_5_/moving_variancebatchnorm_2_/moving_variancebatchnorm_1_/moving_variancebatchnorm_2_/moving_meanbatchnorm_4_/moving_meanbatchnorm_5_/moving_meanbatchnorm_4_/moving_variancebatchnorm_1_/moving_meanbatchnorm_3_/moving_meanbatchnorm_3_/moving_varianceConst_2*\
TinU
S2Q*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *&
f!R
__inference__traced_save_4035
�
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenameimageinput_/meanimageinput_/varianceimageinput_/countconv_1_/kernel_1conv_1_/bias_1batchnorm_1_/gamma_1batchnorm_1_/beta_1batchnorm_1_/moving_mean_1batchnorm_1_/moving_variance_1conv_2_/kernel_1conv_2_/bias_1batchnorm_2_/gamma_1batchnorm_2_/beta_1batchnorm_2_/moving_mean_1batchnorm_2_/moving_variance_1conv_3_/kernel_1conv_3_/bias_1batchnorm_3_/gamma_1batchnorm_3_/beta_1batchnorm_3_/moving_mean_1batchnorm_3_/moving_variance_1conv_4_/kernel_1conv_4_/bias_1batchnorm_4_/gamma_1batchnorm_4_/beta_1batchnorm_4_/moving_mean_1batchnorm_4_/moving_variance_1conv_5_/kernel_1conv_5_/bias_1batchnorm_5_/gamma_1batchnorm_5_/beta_1batchnorm_5_/moving_mean_1batchnorm_5_/moving_variance_1lstm_/lstm_cell/kernel_1"lstm_/lstm_cell/recurrent_kernel_1lstm_/lstm_cell/bias_1seed_generator_state_1fc_1_/kernel_1fc_1_/bias_1seed_generator_statefc_2_/kernel_1fc_2_/bias_1batchnorm_1_/betaconv_2_/kernelbatchnorm_3_/betaconv_4_/kernelconv_5_/biasconv_1_/kernelconv_3_/biasconv_5_/kernelconv_3_/kernellstm_/lstm_cell/kernelfc_1_/kernelconv_1_/biasbatchnorm_2_/gammabatchnorm_4_/gammabatchnorm_5_/gamma lstm_/lstm_cell/recurrent_kernelfc_2_/kernellstm_/lstm_cell/bias
fc_1_/biasbatchnorm_3_/gamma
fc_2_/biasbatchnorm_2_/betabatchnorm_4_/betabatchnorm_5_/betabatchnorm_1_/gammaconv_2_/biasconv_4_/biasbatchnorm_5_/moving_variancebatchnorm_2_/moving_variancebatchnorm_1_/moving_variancebatchnorm_2_/moving_meanbatchnorm_4_/moving_meanbatchnorm_5_/moving_meanbatchnorm_4_/moving_variancebatchnorm_1_/moving_meanbatchnorm_3_/moving_meanbatchnorm_3_/moving_variance*[
TinT
R2P*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__traced_restore_4281��
�"
�	
"__inference_signature_wrapper_2485	
input
unknown
	unknown_0#
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:$

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:$

unknown_19: 

unknown_20: 

unknown_21: 

unknown_22: 

unknown_23: 

unknown_24: $

unknown_25: P

unknown_26:P

unknown_27:P

unknown_28:P

unknown_29:P

unknown_30:P

unknown_31:
��

unknown_32:
��

unknown_33:	�

unknown_34:	�

unknown_35:

unknown_36:

unknown_37:
identity	

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37*3
Tin,
*2(*
Tout
2	*
_collective_manager_ids
 *E
_output_shapes3
1:���������:���������:���������*G
_read_only_resource_inputs)
'%	
 !"#$%&'*-
config_proto

CPU

GPU 2J 8� *"
fR
__inference___call___1913k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*#
_output_shapes
:���������m

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*#
_output_shapes
:���������q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesr
p: :b2:b2: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:$' 

_user_specified_name2477:$& 

_user_specified_name2475:$% 

_user_specified_name2473:$$ 

_user_specified_name2471:$# 

_user_specified_name2469:$" 

_user_specified_name2467:$! 

_user_specified_name2465:$  

_user_specified_name2463:$ 

_user_specified_name2461:$ 

_user_specified_name2459:$ 

_user_specified_name2457:$ 

_user_specified_name2455:$ 

_user_specified_name2453:$ 

_user_specified_name2451:$ 

_user_specified_name2449:$ 

_user_specified_name2447:$ 

_user_specified_name2445:$ 

_user_specified_name2443:$ 

_user_specified_name2441:$ 

_user_specified_name2439:$ 

_user_specified_name2437:$ 

_user_specified_name2435:$ 

_user_specified_name2433:$ 

_user_specified_name2431:$ 

_user_specified_name2429:$ 

_user_specified_name2427:$ 

_user_specified_name2425:$ 

_user_specified_name2423:$ 

_user_specified_name2421:$
 

_user_specified_name2419:$	 

_user_specified_name2417:$ 

_user_specified_name2415:$ 

_user_specified_name2413:$ 

_user_specified_name2411:$ 

_user_specified_name2409:$ 

_user_specified_name2407:$ 

_user_specified_name2405:LH
&
_output_shapes
:b2

_user_specified_name2403:LH
&
_output_shapes
:b2

_user_specified_name2401:= 9

_output_shapes
: 

_user_specified_nameinput
��
�	
__inference___call___3047	
input
unknown
	unknown_0#
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:$

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:$

unknown_19: 

unknown_20: 

unknown_21: 

unknown_22: 

unknown_23: 

unknown_24: $

unknown_25: P

unknown_26:P

unknown_27:P

unknown_28:P

unknown_29:P

unknown_30:P

unknown_31:
��

unknown_32:
��

unknown_33:	�

unknown_34:	�

unknown_35:

unknown_36:

unknown_37:
identity	

identity_1

identity_2��StatefulPartitionedCall3
ReadFileReadFileinput*
_output_shapes
: ~
	DecodeWav	DecodeWavReadFile:contents:0*!
_output_shapes
:	�}: *
desired_channels*
desired_samples�}k
SqueezeSqueezeDecodeWav:audio:0*
T0*
_output_shapes	
:�}*
squeeze_dims

���������L
	Squeeze_1SqueezeSqueeze:output:0*
T0*
_output_shapes	
:�}P
ShapeConst*
_output_shapes
:*
dtype0*
valueB:�}]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskH
sub/xConst*
_output_shapes
: *
dtype0*
value
B :�}S
subSubsub/x:output:0strided_slice:output:0*
T0*
_output_shapes
: K
	Maximum/xConst*
_output_shapes
: *
dtype0*
value	B : P
MaximumMaximumMaximum/x:output:0sub:z:0*
T0*
_output_shapes
: R
Pad/paddings/0/0Const*
_output_shapes
: *
dtype0*
value	B : l
Pad/paddings/0PackPad/paddings/0/0:output:0Maximum:z:0*
N*
T0*
_output_shapes
:_
Pad/paddingsPackPad/paddings/0:output:0*
N*
T0*
_output_shapes

:[
PadPadSqueeze_1:output:0Pad/paddings:output:0*
T0*
_output_shapes	
:�}Q
l2_normalize/SquareSquarePad:output:0*
T0*
_output_shapes	
:�}\
l2_normalize/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
l2_normalize/SumSuml2_normalize/Square:y:0l2_normalize/Const:output:0*
T0*
_output_shapes
:*
	keep_dims([
l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
l2_normalize/MaximumMaximuml2_normalize/Sum:output:0l2_normalize/Maximum/y:output:0*
T0*
_output_shapes
:Z
l2_normalize/RsqrtRsqrtl2_normalize/Maximum:z:0*
T0*
_output_shapes
:_
l2_normalizeMulPad:output:0l2_normalize/Rsqrt:y:0*
T0*
_output_shapes	
:�}T
stft/frame_lengthConst*
_output_shapes
: *
dtype0*
value
B :�R
stft/frame_stepConst*
_output_shapes
: *
dtype0*
value
B :�R
stft/fft_lengthConst*
_output_shapes
: *
dtype0*
value
B :�Z
stft/frame/axisConst*
_output_shapes
: *
dtype0*
valueB :
���������[
stft/frame/ShapeConst*
_output_shapes
:*
dtype0*
valueB:�}X
stft/frame/Size/ConstConst*
_output_shapes
: *
dtype0*
valueB Q
stft/frame/SizeConst*
_output_shapes
: *
dtype0*
value	B : Z
stft/frame/Size_1/ConstConst*
_output_shapes
: *
dtype0*
valueB S
stft/frame/Size_1Const*
_output_shapes
: *
dtype0*
value	B : R
stft/frame/ConstConst*
_output_shapes
: *
dtype0*
value	B : S
stft/frame/sub/xConst*
_output_shapes
: *
dtype0*
value
B :�}m
stft/frame/subSubstft/frame/sub/x:output:0stft/frame_length:output:0*
T0*
_output_shapes
: n
stft/frame/floordivFloorDivstft/frame/sub:z:0stft/frame_step:output:0*
T0*
_output_shapes
: R
stft/frame/add/xConst*
_output_shapes
: *
dtype0*
value	B :l
stft/frame/addAddV2stft/frame/add/x:output:0stft/frame/floordiv:z:0*
T0*
_output_shapes
: m
stft/frame/MaximumMaximumstft/frame/Const:output:0stft/frame/add:z:0*
T0*
_output_shapes
: V
stft/frame/gcd/ConstConst*
_output_shapes
: *
dtype0*
value	B :PY
stft/frame/floordiv_1/yConst*
_output_shapes
: *
dtype0*
value	B :P�
stft/frame/floordiv_1FloorDivstft/frame_length:output:0 stft/frame/floordiv_1/y:output:0*
T0*
_output_shapes
: Y
stft/frame/floordiv_2/yConst*
_output_shapes
: *
dtype0*
value	B :P~
stft/frame/floordiv_2FloorDivstft/frame_step:output:0 stft/frame/floordiv_2/y:output:0*
T0*
_output_shapes
: ]
stft/frame/concat/values_0Const*
_output_shapes
: *
dtype0*
valueB e
stft/frame/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:�}]
stft/frame/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB X
stft/frame/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
stft/frame/concatConcatV2#stft/frame/concat/values_0:output:0#stft/frame/concat/values_1:output:0#stft/frame/concat/values_2:output:0stft/frame/concat/axis:output:0*
N*
T0*
_output_shapes
:_
stft/frame/concat_1/values_0Const*
_output_shapes
: *
dtype0*
valueB m
stft/frame/concat_1/values_1Const*
_output_shapes
:*
dtype0*
valueB"�   P   _
stft/frame/concat_1/values_2Const*
_output_shapes
: *
dtype0*
valueB Z
stft/frame/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
stft/frame/concat_1ConcatV2%stft/frame/concat_1/values_0:output:0%stft/frame/concat_1/values_1:output:0%stft/frame/concat_1/values_2:output:0!stft/frame/concat_1/axis:output:0*
N*
T0*
_output_shapes
:g
stft/frame/zeros_like/tensorConst*
_output_shapes
:*
dtype0*
valueB:�}_
stft/frame/zeros_likeConst*
_output_shapes
:*
dtype0*
valueB: t
*stft/frame/ones_like/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:\
stft/frame/ones_like/ConstConst*
_output_shapes
: *
dtype0*
value	B :�
stft/frame/ones_likeFill3stft/frame/ones_like/Shape/shape_as_tensor:output:0#stft/frame/ones_like/Const:output:0*
T0*
_output_shapes
:�
stft/frame/StridedSliceStridedSlicel2_normalize:z:0stft/frame/zeros_like:output:0stft/frame/concat:output:0stft/frame/ones_like:output:0*
Index0*
T0*
_output_shapes	
:�}�
stft/frame/ReshapeReshape stft/frame/StridedSlice:output:0stft/frame/concat_1:output:0*
T0*
_output_shapes
:	�PX
stft/frame/range/startConst*
_output_shapes
: *
dtype0*
value	B : X
stft/frame/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
stft/frame/rangeRangestft/frame/range/start:output:0stft/frame/Maximum:z:0stft/frame/range/delta:output:0*
_output_shapes
:bp
stft/frame/mulMulstft/frame/range:output:0stft/frame/floordiv_2:z:0*
T0*
_output_shapes
:b^
stft/frame/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :�
stft/frame/Reshape_1/shapePackstft/frame/Maximum:z:0%stft/frame/Reshape_1/shape/1:output:0*
N*
T0*
_output_shapes
:�
stft/frame/Reshape_1Reshapestft/frame/mul:z:0#stft/frame/Reshape_1/shape:output:0*
T0*
_output_shapes

:bZ
stft/frame/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : Z
stft/frame/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
stft/frame/range_1Range!stft/frame/range_1/start:output:0stft/frame/floordiv_1:z:0!stft/frame/range_1/delta:output:0*
_output_shapes
:^
stft/frame/Reshape_2/shape/0Const*
_output_shapes
: *
dtype0*
value	B :�
stft/frame/Reshape_2/shapePack%stft/frame/Reshape_2/shape/0:output:0stft/frame/floordiv_1:z:0*
N*
T0*
_output_shapes
:�
stft/frame/Reshape_2Reshapestft/frame/range_1:output:0#stft/frame/Reshape_2/shape:output:0*
T0*
_output_shapes

:�
stft/frame/add_1AddV2stft/frame/Reshape_1:output:0stft/frame/Reshape_2:output:0*
T0*
_output_shapes

:bU
stft/frame/Const_1Const*
_output_shapes
: *
dtype0*
valueB U
stft/frame/Const_2Const*
_output_shapes
: *
dtype0*
valueB {
stft/frame/packedPackstft/frame/Maximum:z:0stft/frame_length:output:0*
N*
T0*
_output_shapes
:Z
stft/frame/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
stft/frame/GatherV2GatherV2stft/frame/Reshape:output:0stft/frame/add_1:z:0!stft/frame/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*"
_output_shapes
:bPZ
stft/frame/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
stft/frame/concat_2ConcatV2stft/frame/Const_1:output:0stft/frame/packed:output:0stft/frame/Const_2:output:0!stft/frame/concat_2/axis:output:0*
N*
T0*
_output_shapes
:�
stft/frame/Reshape_3Reshapestft/frame/GatherV2:output:0stft/frame/concat_2:output:0*
T0*
_output_shapes
:	b�[
stft/hann_window/periodicConst*
_output_shapes
: *
dtype0
*
value	B
 Zq
stft/hann_window/CastCast"stft/hann_window/periodic:output:0*

DstT0*

SrcT0
*
_output_shapes
: X
stft/hann_window/mod/yConst*
_output_shapes
: *
dtype0*
value	B :~
stft/hann_window/modFloorModstft/frame_length:output:0stft/hann_window/mod/y:output:0*
T0*
_output_shapes
: X
stft/hann_window/sub/xConst*
_output_shapes
: *
dtype0*
value	B :w
stft/hann_window/subSubstft/hann_window/sub/x:output:0stft/hann_window/mod:z:0*
T0*
_output_shapes
: q
stft/hann_window/mulMulstft/hann_window/Cast:y:0stft/hann_window/sub:z:0*
T0*
_output_shapes
: t
stft/hann_window/addAddV2stft/frame_length:output:0stft/hann_window/mul:z:0*
T0*
_output_shapes
: Z
stft/hann_window/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :{
stft/hann_window/sub_1Substft/hann_window/add:z:0!stft/hann_window/sub_1/y:output:0*
T0*
_output_shapes
: k
stft/hann_window/Cast_1Caststft/hann_window/sub_1:z:0*

DstT0*

SrcT0*
_output_shapes
: ^
stft/hann_window/range/startConst*
_output_shapes
: *
dtype0*
value	B : ^
stft/hann_window/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
stft/hann_window/rangeRange%stft/hann_window/range/start:output:0stft/frame_length:output:0%stft/hann_window/range/delta:output:0*
_output_shapes	
:�u
stft/hann_window/Cast_2Caststft/hann_window/range:output:0*

DstT0*

SrcT0*
_output_shapes	
:�[
stft/hann_window/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *��@�
stft/hann_window/mul_1Mulstft/hann_window/Const:output:0stft/hann_window/Cast_2:y:0*
T0*
_output_shapes	
:��
stft/hann_window/truedivRealDivstft/hann_window/mul_1:z:0stft/hann_window/Cast_1:y:0*
T0*
_output_shapes	
:�_
stft/hann_window/CosCosstft/hann_window/truediv:z:0*
T0*
_output_shapes	
:�]
stft/hann_window/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
stft/hann_window/mul_2Mul!stft/hann_window/mul_2/x:output:0stft/hann_window/Cos:y:0*
T0*
_output_shapes	
:�]
stft/hann_window/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
stft/hann_window/sub_2Sub!stft/hann_window/sub_2/x:output:0stft/hann_window/mul_2:z:0*
T0*
_output_shapes	
:�t
stft/mulMulstft/frame/Reshape_3:output:0stft/hann_window/sub_2:z:0*
T0*
_output_shapes
:	b�`
stft/rfft/packedPackstft/fft_length:output:0*
N*
T0*
_output_shapes
:w
stft/rfft/Pad/paddingsConst*
_output_shapes

:*
dtype0*)
value B"            p   m
stft/rfft/PadPadstft/mul:z:0stft/rfft/Pad/paddings:output:0*
T0*
_output_shapes
:	b�_
stft/rfft/fft_lengthConst*
_output_shapes
:*
dtype0*
valueB:�i
	stft/rfftRFFTstft/rfft/Pad:output:0stft/rfft/fft_length:output:0*
_output_shapes
:	b�F
Abs
ComplexAbsstft/rfft:output:0*
_output_shapes
:	b�X
Shape_1Const*
_output_shapes
:*
dtype0*
valueB"b     h
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
)linear_to_mel_weight_matrix/sample_rate/xConst*
_output_shapes
: *
dtype0*
value
B :�}�
'linear_to_mel_weight_matrix/sample_rateCast2linear_to_mel_weight_matrix/sample_rate/x:output:0*

DstT0*

SrcT0*
_output_shapes
: q
,linear_to_mel_weight_matrix/lower_edge_hertzConst*
_output_shapes
: *
dtype0*
valueB
 *  �Bq
,linear_to_mel_weight_matrix/upper_edge_hertzConst*
_output_shapes
: *
dtype0*
valueB
 * �mEf
!linear_to_mel_weight_matrix/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    j
%linear_to_mel_weight_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
#linear_to_mel_weight_matrix/truedivRealDiv+linear_to_mel_weight_matrix/sample_rate:y:0.linear_to_mel_weight_matrix/truediv/y:output:0*
T0*
_output_shapes
: {
)linear_to_mel_weight_matrix/linspace/CastCaststrided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: �
+linear_to_mel_weight_matrix/linspace/Cast_1Cast-linear_to_mel_weight_matrix/linspace/Cast:y:0*

DstT0*

SrcT0*
_output_shapes
: m
*linear_to_mel_weight_matrix/linspace/ShapeConst*
_output_shapes
: *
dtype0*
valueB o
,linear_to_mel_weight_matrix/linspace/Shape_1Const*
_output_shapes
: *
dtype0*
valueB �
2linear_to_mel_weight_matrix/linspace/BroadcastArgsBroadcastArgs3linear_to_mel_weight_matrix/linspace/Shape:output:05linear_to_mel_weight_matrix/linspace/Shape_1:output:0*
_output_shapes
: �
0linear_to_mel_weight_matrix/linspace/BroadcastToBroadcastTo*linear_to_mel_weight_matrix/Const:output:07linear_to_mel_weight_matrix/linspace/BroadcastArgs:r0:0*
T0*
_output_shapes
: �
2linear_to_mel_weight_matrix/linspace/BroadcastTo_1BroadcastTo'linear_to_mel_weight_matrix/truediv:z:07linear_to_mel_weight_matrix/linspace/BroadcastArgs:r0:0*
T0*
_output_shapes
: u
3linear_to_mel_weight_matrix/linspace/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : �
/linear_to_mel_weight_matrix/linspace/ExpandDims
ExpandDims9linear_to_mel_weight_matrix/linspace/BroadcastTo:output:0<linear_to_mel_weight_matrix/linspace/ExpandDims/dim:output:0*
T0*
_output_shapes
:w
5linear_to_mel_weight_matrix/linspace/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
1linear_to_mel_weight_matrix/linspace/ExpandDims_1
ExpandDims;linear_to_mel_weight_matrix/linspace/BroadcastTo_1:output:0>linear_to_mel_weight_matrix/linspace/ExpandDims_1/dim:output:0*
T0*
_output_shapes
:v
,linear_to_mel_weight_matrix/linspace/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:v
,linear_to_mel_weight_matrix/linspace/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:�
8linear_to_mel_weight_matrix/linspace/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
:linear_to_mel_weight_matrix/linspace/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
:linear_to_mel_weight_matrix/linspace/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
2linear_to_mel_weight_matrix/linspace/strided_sliceStridedSlice5linear_to_mel_weight_matrix/linspace/Shape_3:output:0Alinear_to_mel_weight_matrix/linspace/strided_slice/stack:output:0Clinear_to_mel_weight_matrix/linspace/strided_slice/stack_1:output:0Clinear_to_mel_weight_matrix/linspace/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
*linear_to_mel_weight_matrix/linspace/add/yConst*
_output_shapes
: *
dtype0*
value	B : �
(linear_to_mel_weight_matrix/linspace/addAddV2;linear_to_mel_weight_matrix/linspace/strided_slice:output:03linear_to_mel_weight_matrix/linspace/add/y:output:0*
T0*
_output_shapes
: y
7linear_to_mel_weight_matrix/linspace/SelectV2/conditionConst*
_output_shapes
: *
dtype0
*
value	B
 Zq
/linear_to_mel_weight_matrix/linspace/SelectV2/tConst*
_output_shapes
: *
dtype0*
value	B : �
-linear_to_mel_weight_matrix/linspace/SelectV2SelectV2@linear_to_mel_weight_matrix/linspace/SelectV2/condition:output:08linear_to_mel_weight_matrix/linspace/SelectV2/t:output:0,linear_to_mel_weight_matrix/linspace/add:z:0*
T0*
_output_shapes
: l
*linear_to_mel_weight_matrix/linspace/sub/yConst*
_output_shapes
: *
dtype0*
value	B :�
(linear_to_mel_weight_matrix/linspace/subSub-linear_to_mel_weight_matrix/linspace/Cast:y:03linear_to_mel_weight_matrix/linspace/sub/y:output:0*
T0*
_output_shapes
: p
.linear_to_mel_weight_matrix/linspace/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B : �
,linear_to_mel_weight_matrix/linspace/MaximumMaximum,linear_to_mel_weight_matrix/linspace/sub:z:07linear_to_mel_weight_matrix/linspace/Maximum/y:output:0*
T0*
_output_shapes
: n
,linear_to_mel_weight_matrix/linspace/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
*linear_to_mel_weight_matrix/linspace/sub_1Sub-linear_to_mel_weight_matrix/linspace/Cast:y:05linear_to_mel_weight_matrix/linspace/sub_1/y:output:0*
T0*
_output_shapes
: r
0linear_to_mel_weight_matrix/linspace/Maximum_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
.linear_to_mel_weight_matrix/linspace/Maximum_1Maximum.linear_to_mel_weight_matrix/linspace/sub_1:z:09linear_to_mel_weight_matrix/linspace/Maximum_1/y:output:0*
T0*
_output_shapes
: �
*linear_to_mel_weight_matrix/linspace/sub_2Sub:linear_to_mel_weight_matrix/linspace/ExpandDims_1:output:08linear_to_mel_weight_matrix/linspace/ExpandDims:output:0*
T0*
_output_shapes
:�
+linear_to_mel_weight_matrix/linspace/Cast_2Cast2linear_to_mel_weight_matrix/linspace/Maximum_1:z:0*

DstT0*

SrcT0*
_output_shapes
: �
,linear_to_mel_weight_matrix/linspace/truedivRealDiv.linear_to_mel_weight_matrix/linspace/sub_2:z:0/linear_to_mel_weight_matrix/linspace/Cast_2:y:0*
T0*
_output_shapes
:u
3linear_to_mel_weight_matrix/linspace/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : �
1linear_to_mel_weight_matrix/linspace/GreaterEqualGreaterEqual-linear_to_mel_weight_matrix/linspace/Cast:y:0<linear_to_mel_weight_matrix/linspace/GreaterEqual/y:output:0*
T0*
_output_shapes
: |
1linear_to_mel_weight_matrix/linspace/SelectV2_1/eConst*
_output_shapes
: *
dtype0*
valueB :
����������
/linear_to_mel_weight_matrix/linspace/SelectV2_1SelectV25linear_to_mel_weight_matrix/linspace/GreaterEqual:z:02linear_to_mel_weight_matrix/linspace/Maximum_1:z:0:linear_to_mel_weight_matrix/linspace/SelectV2_1/e:output:0*
T0*
_output_shapes
: r
0linear_to_mel_weight_matrix/linspace/range/startConst*
_output_shapes
: *
dtype0	*
value	B	 Rr
0linear_to_mel_weight_matrix/linspace/range/deltaConst*
_output_shapes
: *
dtype0	*
value	B	 R�
/linear_to_mel_weight_matrix/linspace/range/CastCast8linear_to_mel_weight_matrix/linspace/SelectV2_1:output:0*

DstT0	*

SrcT0*
_output_shapes
: �
*linear_to_mel_weight_matrix/linspace/rangeRange9linear_to_mel_weight_matrix/linspace/range/start:output:03linear_to_mel_weight_matrix/linspace/range/Cast:y:09linear_to_mel_weight_matrix/linspace/range/delta:output:0*

Tidx0	*
_output_shapes	
:��
+linear_to_mel_weight_matrix/linspace/Cast_3Cast3linear_to_mel_weight_matrix/linspace/range:output:0*

DstT0*

SrcT0	*
_output_shapes	
:�t
2linear_to_mel_weight_matrix/linspace/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : t
2linear_to_mel_weight_matrix/linspace/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
,linear_to_mel_weight_matrix/linspace/range_1Range;linear_to_mel_weight_matrix/linspace/range_1/start:output:0;linear_to_mel_weight_matrix/linspace/strided_slice:output:0;linear_to_mel_weight_matrix/linspace/range_1/delta:output:0*
_output_shapes
:�
*linear_to_mel_weight_matrix/linspace/EqualEqual6linear_to_mel_weight_matrix/linspace/SelectV2:output:05linear_to_mel_weight_matrix/linspace/range_1:output:0*
T0*
_output_shapes
:s
1linear_to_mel_weight_matrix/linspace/SelectV2_2/eConst*
_output_shapes
: *
dtype0*
value	B :�
/linear_to_mel_weight_matrix/linspace/SelectV2_2SelectV2.linear_to_mel_weight_matrix/linspace/Equal:z:00linear_to_mel_weight_matrix/linspace/Maximum:z:0:linear_to_mel_weight_matrix/linspace/SelectV2_2/e:output:0*
T0*
_output_shapes
:�
,linear_to_mel_weight_matrix/linspace/ReshapeReshape/linear_to_mel_weight_matrix/linspace/Cast_3:y:08linear_to_mel_weight_matrix/linspace/SelectV2_2:output:0*
T0*
_output_shapes	
:��
(linear_to_mel_weight_matrix/linspace/mulMul0linear_to_mel_weight_matrix/linspace/truediv:z:05linear_to_mel_weight_matrix/linspace/Reshape:output:0*
T0*
_output_shapes	
:��
*linear_to_mel_weight_matrix/linspace/add_1AddV28linear_to_mel_weight_matrix/linspace/ExpandDims:output:0,linear_to_mel_weight_matrix/linspace/mul:z:0*
T0*
_output_shapes	
:��
+linear_to_mel_weight_matrix/linspace/concatConcatV28linear_to_mel_weight_matrix/linspace/ExpandDims:output:0.linear_to_mel_weight_matrix/linspace/add_1:z:0:linear_to_mel_weight_matrix/linspace/ExpandDims_1:output:06linear_to_mel_weight_matrix/linspace/SelectV2:output:0*
N*
T0*
_output_shapes	
:�y
/linear_to_mel_weight_matrix/linspace/zeros_likeConst*
_output_shapes
:*
dtype0*
valueB: l
*linear_to_mel_weight_matrix/linspace/ConstConst*
_output_shapes
: *
dtype0*
value	B :�
:linear_to_mel_weight_matrix/linspace/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
<linear_to_mel_weight_matrix/linspace/strided_slice_1/stack_1Pack6linear_to_mel_weight_matrix/linspace/SelectV2:output:0*
N*
T0*
_output_shapes
:�
<linear_to_mel_weight_matrix/linspace/strided_slice_1/stack_2Pack3linear_to_mel_weight_matrix/linspace/Const:output:0*
N*
T0*
_output_shapes
:�
4linear_to_mel_weight_matrix/linspace/strided_slice_1StridedSlice5linear_to_mel_weight_matrix/linspace/Shape_2:output:0Clinear_to_mel_weight_matrix/linspace/strided_slice_1/stack:output:0Elinear_to_mel_weight_matrix/linspace/strided_slice_1/stack_1:output:0Elinear_to_mel_weight_matrix/linspace/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: ~
4linear_to_mel_weight_matrix/linspace/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:�
.linear_to_mel_weight_matrix/linspace/Reshape_1Reshape-linear_to_mel_weight_matrix/linspace/Cast:y:0=linear_to_mel_weight_matrix/linspace/Reshape_1/shape:output:0*
T0*
_output_shapes
:n
,linear_to_mel_weight_matrix/linspace/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :�
*linear_to_mel_weight_matrix/linspace/add_2AddV26linear_to_mel_weight_matrix/linspace/SelectV2:output:05linear_to_mel_weight_matrix/linspace/add_2/y:output:0*
T0*
_output_shapes
: n
,linear_to_mel_weight_matrix/linspace/Const_1Const*
_output_shapes
: *
dtype0*
value	B : n
,linear_to_mel_weight_matrix/linspace/Const_2Const*
_output_shapes
: *
dtype0*
value	B :�
:linear_to_mel_weight_matrix/linspace/strided_slice_2/stackPack.linear_to_mel_weight_matrix/linspace/add_2:z:0*
N*
T0*
_output_shapes
:�
<linear_to_mel_weight_matrix/linspace/strided_slice_2/stack_1Pack5linear_to_mel_weight_matrix/linspace/Const_1:output:0*
N*
T0*
_output_shapes
:�
<linear_to_mel_weight_matrix/linspace/strided_slice_2/stack_2Pack5linear_to_mel_weight_matrix/linspace/Const_2:output:0*
N*
T0*
_output_shapes
:�
4linear_to_mel_weight_matrix/linspace/strided_slice_2StridedSlice5linear_to_mel_weight_matrix/linspace/Shape_2:output:0Clinear_to_mel_weight_matrix/linspace/strided_slice_2/stack:output:0Elinear_to_mel_weight_matrix/linspace/strided_slice_2/stack_1:output:0Elinear_to_mel_weight_matrix/linspace/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_maskt
2linear_to_mel_weight_matrix/linspace/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
-linear_to_mel_weight_matrix/linspace/concat_1ConcatV2=linear_to_mel_weight_matrix/linspace/strided_slice_1:output:07linear_to_mel_weight_matrix/linspace/Reshape_1:output:0=linear_to_mel_weight_matrix/linspace/strided_slice_2:output:0;linear_to_mel_weight_matrix/linspace/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
*linear_to_mel_weight_matrix/linspace/SliceSlice4linear_to_mel_weight_matrix/linspace/concat:output:08linear_to_mel_weight_matrix/linspace/zeros_like:output:06linear_to_mel_weight_matrix/linspace/concat_1:output:0*
Index0*
T0*
_output_shapes	
:�y
/linear_to_mel_weight_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:{
1linear_to_mel_weight_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: {
1linear_to_mel_weight_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
)linear_to_mel_weight_matrix/strided_sliceStridedSlice3linear_to_mel_weight_matrix/linspace/Slice:output:08linear_to_mel_weight_matrix/strided_slice/stack:output:0:linear_to_mel_weight_matrix/strided_slice/stack_1:output:0:linear_to_mel_weight_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes	
:�*
end_maskw
2linear_to_mel_weight_matrix/hertz_to_mel/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  /D�
0linear_to_mel_weight_matrix/hertz_to_mel/truedivRealDiv2linear_to_mel_weight_matrix/strided_slice:output:0;linear_to_mel_weight_matrix/hertz_to_mel/truediv/y:output:0*
T0*
_output_shapes	
:�s
.linear_to_mel_weight_matrix/hertz_to_mel/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
,linear_to_mel_weight_matrix/hertz_to_mel/addAddV27linear_to_mel_weight_matrix/hertz_to_mel/add/x:output:04linear_to_mel_weight_matrix/hertz_to_mel/truediv:z:0*
T0*
_output_shapes	
:��
,linear_to_mel_weight_matrix/hertz_to_mel/LogLog0linear_to_mel_weight_matrix/hertz_to_mel/add:z:0*
T0*
_output_shapes	
:�s
.linear_to_mel_weight_matrix/hertz_to_mel/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 * ��D�
,linear_to_mel_weight_matrix/hertz_to_mel/mulMul7linear_to_mel_weight_matrix/hertz_to_mel/mul/x:output:00linear_to_mel_weight_matrix/hertz_to_mel/Log:y:0*
T0*
_output_shapes	
:�l
*linear_to_mel_weight_matrix/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
&linear_to_mel_weight_matrix/ExpandDims
ExpandDims0linear_to_mel_weight_matrix/hertz_to_mel/mul:z:03linear_to_mel_weight_matrix/ExpandDims/dim:output:0*
T0*
_output_shapes
:	�y
4linear_to_mel_weight_matrix/hertz_to_mel_1/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  /D�
2linear_to_mel_weight_matrix/hertz_to_mel_1/truedivRealDiv5linear_to_mel_weight_matrix/lower_edge_hertz:output:0=linear_to_mel_weight_matrix/hertz_to_mel_1/truediv/y:output:0*
T0*
_output_shapes
: u
0linear_to_mel_weight_matrix/hertz_to_mel_1/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
.linear_to_mel_weight_matrix/hertz_to_mel_1/addAddV29linear_to_mel_weight_matrix/hertz_to_mel_1/add/x:output:06linear_to_mel_weight_matrix/hertz_to_mel_1/truediv:z:0*
T0*
_output_shapes
: �
.linear_to_mel_weight_matrix/hertz_to_mel_1/LogLog2linear_to_mel_weight_matrix/hertz_to_mel_1/add:z:0*
T0*
_output_shapes
: u
0linear_to_mel_weight_matrix/hertz_to_mel_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 * ��D�
.linear_to_mel_weight_matrix/hertz_to_mel_1/mulMul9linear_to_mel_weight_matrix/hertz_to_mel_1/mul/x:output:02linear_to_mel_weight_matrix/hertz_to_mel_1/Log:y:0*
T0*
_output_shapes
: y
4linear_to_mel_weight_matrix/hertz_to_mel_2/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  /D�
2linear_to_mel_weight_matrix/hertz_to_mel_2/truedivRealDiv5linear_to_mel_weight_matrix/upper_edge_hertz:output:0=linear_to_mel_weight_matrix/hertz_to_mel_2/truediv/y:output:0*
T0*
_output_shapes
: u
0linear_to_mel_weight_matrix/hertz_to_mel_2/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
.linear_to_mel_weight_matrix/hertz_to_mel_2/addAddV29linear_to_mel_weight_matrix/hertz_to_mel_2/add/x:output:06linear_to_mel_weight_matrix/hertz_to_mel_2/truediv:z:0*
T0*
_output_shapes
: �
.linear_to_mel_weight_matrix/hertz_to_mel_2/LogLog2linear_to_mel_weight_matrix/hertz_to_mel_2/add:z:0*
T0*
_output_shapes
: u
0linear_to_mel_weight_matrix/hertz_to_mel_2/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 * ��D�
.linear_to_mel_weight_matrix/hertz_to_mel_2/mulMul9linear_to_mel_weight_matrix/hertz_to_mel_2/mul/x:output:02linear_to_mel_weight_matrix/hertz_to_mel_2/Log:y:0*
T0*
_output_shapes
: l
*linear_to_mel_weight_matrix/linspace_1/numConst*
_output_shapes
: *
dtype0*
value	B :4�
+linear_to_mel_weight_matrix/linspace_1/CastCast3linear_to_mel_weight_matrix/linspace_1/num:output:0*

DstT0*

SrcT0*
_output_shapes
: �
-linear_to_mel_weight_matrix/linspace_1/Cast_1Cast/linear_to_mel_weight_matrix/linspace_1/Cast:y:0*

DstT0*

SrcT0*
_output_shapes
: o
,linear_to_mel_weight_matrix/linspace_1/ShapeConst*
_output_shapes
: *
dtype0*
valueB q
.linear_to_mel_weight_matrix/linspace_1/Shape_1Const*
_output_shapes
: *
dtype0*
valueB �
4linear_to_mel_weight_matrix/linspace_1/BroadcastArgsBroadcastArgs5linear_to_mel_weight_matrix/linspace_1/Shape:output:07linear_to_mel_weight_matrix/linspace_1/Shape_1:output:0*
_output_shapes
: �
2linear_to_mel_weight_matrix/linspace_1/BroadcastToBroadcastTo2linear_to_mel_weight_matrix/hertz_to_mel_1/mul:z:09linear_to_mel_weight_matrix/linspace_1/BroadcastArgs:r0:0*
T0*
_output_shapes
: �
4linear_to_mel_weight_matrix/linspace_1/BroadcastTo_1BroadcastTo2linear_to_mel_weight_matrix/hertz_to_mel_2/mul:z:09linear_to_mel_weight_matrix/linspace_1/BroadcastArgs:r0:0*
T0*
_output_shapes
: w
5linear_to_mel_weight_matrix/linspace_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : �
1linear_to_mel_weight_matrix/linspace_1/ExpandDims
ExpandDims;linear_to_mel_weight_matrix/linspace_1/BroadcastTo:output:0>linear_to_mel_weight_matrix/linspace_1/ExpandDims/dim:output:0*
T0*
_output_shapes
:y
7linear_to_mel_weight_matrix/linspace_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
3linear_to_mel_weight_matrix/linspace_1/ExpandDims_1
ExpandDims=linear_to_mel_weight_matrix/linspace_1/BroadcastTo_1:output:0@linear_to_mel_weight_matrix/linspace_1/ExpandDims_1/dim:output:0*
T0*
_output_shapes
:x
.linear_to_mel_weight_matrix/linspace_1/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:x
.linear_to_mel_weight_matrix/linspace_1/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:�
:linear_to_mel_weight_matrix/linspace_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
<linear_to_mel_weight_matrix/linspace_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
<linear_to_mel_weight_matrix/linspace_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
4linear_to_mel_weight_matrix/linspace_1/strided_sliceStridedSlice7linear_to_mel_weight_matrix/linspace_1/Shape_3:output:0Clinear_to_mel_weight_matrix/linspace_1/strided_slice/stack:output:0Elinear_to_mel_weight_matrix/linspace_1/strided_slice/stack_1:output:0Elinear_to_mel_weight_matrix/linspace_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
,linear_to_mel_weight_matrix/linspace_1/add/yConst*
_output_shapes
: *
dtype0*
value	B : �
*linear_to_mel_weight_matrix/linspace_1/addAddV2=linear_to_mel_weight_matrix/linspace_1/strided_slice:output:05linear_to_mel_weight_matrix/linspace_1/add/y:output:0*
T0*
_output_shapes
: {
9linear_to_mel_weight_matrix/linspace_1/SelectV2/conditionConst*
_output_shapes
: *
dtype0
*
value	B
 Zs
1linear_to_mel_weight_matrix/linspace_1/SelectV2/tConst*
_output_shapes
: *
dtype0*
value	B : �
/linear_to_mel_weight_matrix/linspace_1/SelectV2SelectV2Blinear_to_mel_weight_matrix/linspace_1/SelectV2/condition:output:0:linear_to_mel_weight_matrix/linspace_1/SelectV2/t:output:0.linear_to_mel_weight_matrix/linspace_1/add:z:0*
T0*
_output_shapes
: n
,linear_to_mel_weight_matrix/linspace_1/sub/yConst*
_output_shapes
: *
dtype0*
value	B :�
*linear_to_mel_weight_matrix/linspace_1/subSub/linear_to_mel_weight_matrix/linspace_1/Cast:y:05linear_to_mel_weight_matrix/linspace_1/sub/y:output:0*
T0*
_output_shapes
: r
0linear_to_mel_weight_matrix/linspace_1/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B : �
.linear_to_mel_weight_matrix/linspace_1/MaximumMaximum.linear_to_mel_weight_matrix/linspace_1/sub:z:09linear_to_mel_weight_matrix/linspace_1/Maximum/y:output:0*
T0*
_output_shapes
: p
.linear_to_mel_weight_matrix/linspace_1/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
,linear_to_mel_weight_matrix/linspace_1/sub_1Sub/linear_to_mel_weight_matrix/linspace_1/Cast:y:07linear_to_mel_weight_matrix/linspace_1/sub_1/y:output:0*
T0*
_output_shapes
: t
2linear_to_mel_weight_matrix/linspace_1/Maximum_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
0linear_to_mel_weight_matrix/linspace_1/Maximum_1Maximum0linear_to_mel_weight_matrix/linspace_1/sub_1:z:0;linear_to_mel_weight_matrix/linspace_1/Maximum_1/y:output:0*
T0*
_output_shapes
: �
,linear_to_mel_weight_matrix/linspace_1/sub_2Sub<linear_to_mel_weight_matrix/linspace_1/ExpandDims_1:output:0:linear_to_mel_weight_matrix/linspace_1/ExpandDims:output:0*
T0*
_output_shapes
:�
-linear_to_mel_weight_matrix/linspace_1/Cast_2Cast4linear_to_mel_weight_matrix/linspace_1/Maximum_1:z:0*

DstT0*

SrcT0*
_output_shapes
: �
.linear_to_mel_weight_matrix/linspace_1/truedivRealDiv0linear_to_mel_weight_matrix/linspace_1/sub_2:z:01linear_to_mel_weight_matrix/linspace_1/Cast_2:y:0*
T0*
_output_shapes
:w
5linear_to_mel_weight_matrix/linspace_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : �
3linear_to_mel_weight_matrix/linspace_1/GreaterEqualGreaterEqual/linear_to_mel_weight_matrix/linspace_1/Cast:y:0>linear_to_mel_weight_matrix/linspace_1/GreaterEqual/y:output:0*
T0*
_output_shapes
: ~
3linear_to_mel_weight_matrix/linspace_1/SelectV2_1/eConst*
_output_shapes
: *
dtype0*
valueB :
����������
1linear_to_mel_weight_matrix/linspace_1/SelectV2_1SelectV27linear_to_mel_weight_matrix/linspace_1/GreaterEqual:z:04linear_to_mel_weight_matrix/linspace_1/Maximum_1:z:0<linear_to_mel_weight_matrix/linspace_1/SelectV2_1/e:output:0*
T0*
_output_shapes
: t
2linear_to_mel_weight_matrix/linspace_1/range/startConst*
_output_shapes
: *
dtype0	*
value	B	 Rt
2linear_to_mel_weight_matrix/linspace_1/range/deltaConst*
_output_shapes
: *
dtype0	*
value	B	 R�
1linear_to_mel_weight_matrix/linspace_1/range/CastCast:linear_to_mel_weight_matrix/linspace_1/SelectV2_1:output:0*

DstT0	*

SrcT0*
_output_shapes
: �
,linear_to_mel_weight_matrix/linspace_1/rangeRange;linear_to_mel_weight_matrix/linspace_1/range/start:output:05linear_to_mel_weight_matrix/linspace_1/range/Cast:y:0;linear_to_mel_weight_matrix/linspace_1/range/delta:output:0*

Tidx0	*
_output_shapes
:2�
-linear_to_mel_weight_matrix/linspace_1/Cast_3Cast5linear_to_mel_weight_matrix/linspace_1/range:output:0*

DstT0*

SrcT0	*
_output_shapes
:2v
4linear_to_mel_weight_matrix/linspace_1/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : v
4linear_to_mel_weight_matrix/linspace_1/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
.linear_to_mel_weight_matrix/linspace_1/range_1Range=linear_to_mel_weight_matrix/linspace_1/range_1/start:output:0=linear_to_mel_weight_matrix/linspace_1/strided_slice:output:0=linear_to_mel_weight_matrix/linspace_1/range_1/delta:output:0*
_output_shapes
:�
,linear_to_mel_weight_matrix/linspace_1/EqualEqual8linear_to_mel_weight_matrix/linspace_1/SelectV2:output:07linear_to_mel_weight_matrix/linspace_1/range_1:output:0*
T0*
_output_shapes
:u
3linear_to_mel_weight_matrix/linspace_1/SelectV2_2/eConst*
_output_shapes
: *
dtype0*
value	B :�
1linear_to_mel_weight_matrix/linspace_1/SelectV2_2SelectV20linear_to_mel_weight_matrix/linspace_1/Equal:z:02linear_to_mel_weight_matrix/linspace_1/Maximum:z:0<linear_to_mel_weight_matrix/linspace_1/SelectV2_2/e:output:0*
T0*
_output_shapes
:�
.linear_to_mel_weight_matrix/linspace_1/ReshapeReshape1linear_to_mel_weight_matrix/linspace_1/Cast_3:y:0:linear_to_mel_weight_matrix/linspace_1/SelectV2_2:output:0*
T0*
_output_shapes
:2�
*linear_to_mel_weight_matrix/linspace_1/mulMul2linear_to_mel_weight_matrix/linspace_1/truediv:z:07linear_to_mel_weight_matrix/linspace_1/Reshape:output:0*
T0*
_output_shapes
:2�
,linear_to_mel_weight_matrix/linspace_1/add_1AddV2:linear_to_mel_weight_matrix/linspace_1/ExpandDims:output:0.linear_to_mel_weight_matrix/linspace_1/mul:z:0*
T0*
_output_shapes
:2�
-linear_to_mel_weight_matrix/linspace_1/concatConcatV2:linear_to_mel_weight_matrix/linspace_1/ExpandDims:output:00linear_to_mel_weight_matrix/linspace_1/add_1:z:0<linear_to_mel_weight_matrix/linspace_1/ExpandDims_1:output:08linear_to_mel_weight_matrix/linspace_1/SelectV2:output:0*
N*
T0*
_output_shapes
:4{
1linear_to_mel_weight_matrix/linspace_1/zeros_likeConst*
_output_shapes
:*
dtype0*
valueB: n
,linear_to_mel_weight_matrix/linspace_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :�
<linear_to_mel_weight_matrix/linspace_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
>linear_to_mel_weight_matrix/linspace_1/strided_slice_1/stack_1Pack8linear_to_mel_weight_matrix/linspace_1/SelectV2:output:0*
N*
T0*
_output_shapes
:�
>linear_to_mel_weight_matrix/linspace_1/strided_slice_1/stack_2Pack5linear_to_mel_weight_matrix/linspace_1/Const:output:0*
N*
T0*
_output_shapes
:�
6linear_to_mel_weight_matrix/linspace_1/strided_slice_1StridedSlice7linear_to_mel_weight_matrix/linspace_1/Shape_2:output:0Elinear_to_mel_weight_matrix/linspace_1/strided_slice_1/stack:output:0Glinear_to_mel_weight_matrix/linspace_1/strided_slice_1/stack_1:output:0Glinear_to_mel_weight_matrix/linspace_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: �
6linear_to_mel_weight_matrix/linspace_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:�
0linear_to_mel_weight_matrix/linspace_1/Reshape_1Reshape/linear_to_mel_weight_matrix/linspace_1/Cast:y:0?linear_to_mel_weight_matrix/linspace_1/Reshape_1/shape:output:0*
T0*
_output_shapes
:p
.linear_to_mel_weight_matrix/linspace_1/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :�
,linear_to_mel_weight_matrix/linspace_1/add_2AddV28linear_to_mel_weight_matrix/linspace_1/SelectV2:output:07linear_to_mel_weight_matrix/linspace_1/add_2/y:output:0*
T0*
_output_shapes
: p
.linear_to_mel_weight_matrix/linspace_1/Const_1Const*
_output_shapes
: *
dtype0*
value	B : p
.linear_to_mel_weight_matrix/linspace_1/Const_2Const*
_output_shapes
: *
dtype0*
value	B :�
<linear_to_mel_weight_matrix/linspace_1/strided_slice_2/stackPack0linear_to_mel_weight_matrix/linspace_1/add_2:z:0*
N*
T0*
_output_shapes
:�
>linear_to_mel_weight_matrix/linspace_1/strided_slice_2/stack_1Pack7linear_to_mel_weight_matrix/linspace_1/Const_1:output:0*
N*
T0*
_output_shapes
:�
>linear_to_mel_weight_matrix/linspace_1/strided_slice_2/stack_2Pack7linear_to_mel_weight_matrix/linspace_1/Const_2:output:0*
N*
T0*
_output_shapes
:�
6linear_to_mel_weight_matrix/linspace_1/strided_slice_2StridedSlice7linear_to_mel_weight_matrix/linspace_1/Shape_2:output:0Elinear_to_mel_weight_matrix/linspace_1/strided_slice_2/stack:output:0Glinear_to_mel_weight_matrix/linspace_1/strided_slice_2/stack_1:output:0Glinear_to_mel_weight_matrix/linspace_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_maskv
4linear_to_mel_weight_matrix/linspace_1/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
/linear_to_mel_weight_matrix/linspace_1/concat_1ConcatV2?linear_to_mel_weight_matrix/linspace_1/strided_slice_1:output:09linear_to_mel_weight_matrix/linspace_1/Reshape_1:output:0?linear_to_mel_weight_matrix/linspace_1/strided_slice_2:output:0=linear_to_mel_weight_matrix/linspace_1/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
,linear_to_mel_weight_matrix/linspace_1/SliceSlice6linear_to_mel_weight_matrix/linspace_1/concat:output:0:linear_to_mel_weight_matrix/linspace_1/zeros_like:output:08linear_to_mel_weight_matrix/linspace_1/concat_1:output:0*
Index0*
T0*
_output_shapes
:4p
.linear_to_mel_weight_matrix/frame/frame_lengthConst*
_output_shapes
: *
dtype0*
value	B :n
,linear_to_mel_weight_matrix/frame/frame_stepConst*
_output_shapes
: *
dtype0*
value	B :q
&linear_to_mel_weight_matrix/frame/axisConst*
_output_shapes
: *
dtype0*
valueB :
���������q
'linear_to_mel_weight_matrix/frame/ShapeConst*
_output_shapes
:*
dtype0*
valueB:4o
,linear_to_mel_weight_matrix/frame/Size/ConstConst*
_output_shapes
: *
dtype0*
valueB h
&linear_to_mel_weight_matrix/frame/SizeConst*
_output_shapes
: *
dtype0*
value	B : q
.linear_to_mel_weight_matrix/frame/Size_1/ConstConst*
_output_shapes
: *
dtype0*
valueB j
(linear_to_mel_weight_matrix/frame/Size_1Const*
_output_shapes
: *
dtype0*
value	B : i
'linear_to_mel_weight_matrix/frame/ConstConst*
_output_shapes
: *
dtype0*
value	B : i
'linear_to_mel_weight_matrix/frame/sub/xConst*
_output_shapes
: *
dtype0*
value	B :4�
%linear_to_mel_weight_matrix/frame/subSub0linear_to_mel_weight_matrix/frame/sub/x:output:07linear_to_mel_weight_matrix/frame/frame_length:output:0*
T0*
_output_shapes
: �
*linear_to_mel_weight_matrix/frame/floordivFloorDiv)linear_to_mel_weight_matrix/frame/sub:z:05linear_to_mel_weight_matrix/frame/frame_step:output:0*
T0*
_output_shapes
: i
'linear_to_mel_weight_matrix/frame/add/xConst*
_output_shapes
: *
dtype0*
value	B :�
%linear_to_mel_weight_matrix/frame/addAddV20linear_to_mel_weight_matrix/frame/add/x:output:0.linear_to_mel_weight_matrix/frame/floordiv:z:0*
T0*
_output_shapes
: �
)linear_to_mel_weight_matrix/frame/MaximumMaximum0linear_to_mel_weight_matrix/frame/Const:output:0)linear_to_mel_weight_matrix/frame/add:z:0*
T0*
_output_shapes
: m
+linear_to_mel_weight_matrix/frame/gcd/ConstConst*
_output_shapes
: *
dtype0*
value	B :p
.linear_to_mel_weight_matrix/frame/floordiv_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
,linear_to_mel_weight_matrix/frame/floordiv_1FloorDiv7linear_to_mel_weight_matrix/frame/frame_length:output:07linear_to_mel_weight_matrix/frame/floordiv_1/y:output:0*
T0*
_output_shapes
: p
.linear_to_mel_weight_matrix/frame/floordiv_2/yConst*
_output_shapes
: *
dtype0*
value	B :�
,linear_to_mel_weight_matrix/frame/floordiv_2FloorDiv5linear_to_mel_weight_matrix/frame/frame_step:output:07linear_to_mel_weight_matrix/frame/floordiv_2/y:output:0*
T0*
_output_shapes
: t
1linear_to_mel_weight_matrix/frame/concat/values_0Const*
_output_shapes
: *
dtype0*
valueB {
1linear_to_mel_weight_matrix/frame/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:4t
1linear_to_mel_weight_matrix/frame/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB o
-linear_to_mel_weight_matrix/frame/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
(linear_to_mel_weight_matrix/frame/concatConcatV2:linear_to_mel_weight_matrix/frame/concat/values_0:output:0:linear_to_mel_weight_matrix/frame/concat/values_1:output:0:linear_to_mel_weight_matrix/frame/concat/values_2:output:06linear_to_mel_weight_matrix/frame/concat/axis:output:0*
N*
T0*
_output_shapes
:v
3linear_to_mel_weight_matrix/frame/concat_1/values_0Const*
_output_shapes
: *
dtype0*
valueB �
3linear_to_mel_weight_matrix/frame/concat_1/values_1Const*
_output_shapes
:*
dtype0*
valueB"4      v
3linear_to_mel_weight_matrix/frame/concat_1/values_2Const*
_output_shapes
: *
dtype0*
valueB q
/linear_to_mel_weight_matrix/frame/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
*linear_to_mel_weight_matrix/frame/concat_1ConcatV2<linear_to_mel_weight_matrix/frame/concat_1/values_0:output:0<linear_to_mel_weight_matrix/frame/concat_1/values_1:output:0<linear_to_mel_weight_matrix/frame/concat_1/values_2:output:08linear_to_mel_weight_matrix/frame/concat_1/axis:output:0*
N*
T0*
_output_shapes
:}
3linear_to_mel_weight_matrix/frame/zeros_like/tensorConst*
_output_shapes
:*
dtype0*
valueB:4v
,linear_to_mel_weight_matrix/frame/zeros_likeConst*
_output_shapes
:*
dtype0*
valueB: �
Alinear_to_mel_weight_matrix/frame/ones_like/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:s
1linear_to_mel_weight_matrix/frame/ones_like/ConstConst*
_output_shapes
: *
dtype0*
value	B :�
+linear_to_mel_weight_matrix/frame/ones_likeFillJlinear_to_mel_weight_matrix/frame/ones_like/Shape/shape_as_tensor:output:0:linear_to_mel_weight_matrix/frame/ones_like/Const:output:0*
T0*
_output_shapes
:�
.linear_to_mel_weight_matrix/frame/StridedSliceStridedSlice5linear_to_mel_weight_matrix/linspace_1/Slice:output:05linear_to_mel_weight_matrix/frame/zeros_like:output:01linear_to_mel_weight_matrix/frame/concat:output:04linear_to_mel_weight_matrix/frame/ones_like:output:0*
Index0*
T0*
_output_shapes
:4�
)linear_to_mel_weight_matrix/frame/ReshapeReshape7linear_to_mel_weight_matrix/frame/StridedSlice:output:03linear_to_mel_weight_matrix/frame/concat_1:output:0*
T0*
_output_shapes

:4o
-linear_to_mel_weight_matrix/frame/range/startConst*
_output_shapes
: *
dtype0*
value	B : o
-linear_to_mel_weight_matrix/frame/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
'linear_to_mel_weight_matrix/frame/rangeRange6linear_to_mel_weight_matrix/frame/range/start:output:0-linear_to_mel_weight_matrix/frame/Maximum:z:06linear_to_mel_weight_matrix/frame/range/delta:output:0*
_output_shapes
:2�
%linear_to_mel_weight_matrix/frame/mulMul0linear_to_mel_weight_matrix/frame/range:output:00linear_to_mel_weight_matrix/frame/floordiv_2:z:0*
T0*
_output_shapes
:2u
3linear_to_mel_weight_matrix/frame/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :�
1linear_to_mel_weight_matrix/frame/Reshape_1/shapePack-linear_to_mel_weight_matrix/frame/Maximum:z:0<linear_to_mel_weight_matrix/frame/Reshape_1/shape/1:output:0*
N*
T0*
_output_shapes
:�
+linear_to_mel_weight_matrix/frame/Reshape_1Reshape)linear_to_mel_weight_matrix/frame/mul:z:0:linear_to_mel_weight_matrix/frame/Reshape_1/shape:output:0*
T0*
_output_shapes

:2q
/linear_to_mel_weight_matrix/frame/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : q
/linear_to_mel_weight_matrix/frame/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
)linear_to_mel_weight_matrix/frame/range_1Range8linear_to_mel_weight_matrix/frame/range_1/start:output:00linear_to_mel_weight_matrix/frame/floordiv_1:z:08linear_to_mel_weight_matrix/frame/range_1/delta:output:0*
_output_shapes
:u
3linear_to_mel_weight_matrix/frame/Reshape_2/shape/0Const*
_output_shapes
: *
dtype0*
value	B :�
1linear_to_mel_weight_matrix/frame/Reshape_2/shapePack<linear_to_mel_weight_matrix/frame/Reshape_2/shape/0:output:00linear_to_mel_weight_matrix/frame/floordiv_1:z:0*
N*
T0*
_output_shapes
:�
+linear_to_mel_weight_matrix/frame/Reshape_2Reshape2linear_to_mel_weight_matrix/frame/range_1:output:0:linear_to_mel_weight_matrix/frame/Reshape_2/shape:output:0*
T0*
_output_shapes

:�
'linear_to_mel_weight_matrix/frame/add_1AddV24linear_to_mel_weight_matrix/frame/Reshape_1:output:04linear_to_mel_weight_matrix/frame/Reshape_2:output:0*
T0*
_output_shapes

:2l
)linear_to_mel_weight_matrix/frame/Const_1Const*
_output_shapes
: *
dtype0*
valueB l
)linear_to_mel_weight_matrix/frame/Const_2Const*
_output_shapes
: *
dtype0*
valueB �
(linear_to_mel_weight_matrix/frame/packedPack-linear_to_mel_weight_matrix/frame/Maximum:z:07linear_to_mel_weight_matrix/frame/frame_length:output:0*
N*
T0*
_output_shapes
:q
/linear_to_mel_weight_matrix/frame/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
*linear_to_mel_weight_matrix/frame/GatherV2GatherV22linear_to_mel_weight_matrix/frame/Reshape:output:0+linear_to_mel_weight_matrix/frame/add_1:z:08linear_to_mel_weight_matrix/frame/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*"
_output_shapes
:2q
/linear_to_mel_weight_matrix/frame/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
*linear_to_mel_weight_matrix/frame/concat_2ConcatV22linear_to_mel_weight_matrix/frame/Const_1:output:01linear_to_mel_weight_matrix/frame/packed:output:02linear_to_mel_weight_matrix/frame/Const_2:output:08linear_to_mel_weight_matrix/frame/concat_2/axis:output:0*
N*
T0*
_output_shapes
:�
+linear_to_mel_weight_matrix/frame/Reshape_3Reshape3linear_to_mel_weight_matrix/frame/GatherV2:output:03linear_to_mel_weight_matrix/frame/concat_2:output:0*
T0*
_output_shapes

:2m
+linear_to_mel_weight_matrix/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
!linear_to_mel_weight_matrix/splitSplit4linear_to_mel_weight_matrix/split/split_dim:output:04linear_to_mel_weight_matrix/frame/Reshape_3:output:0*
T0*2
_output_shapes 
:2:2:2*
	num_splitz
)linear_to_mel_weight_matrix/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   2   �
#linear_to_mel_weight_matrix/ReshapeReshape*linear_to_mel_weight_matrix/split:output:02linear_to_mel_weight_matrix/Reshape/shape:output:0*
T0*
_output_shapes

:2|
+linear_to_mel_weight_matrix/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   2   �
%linear_to_mel_weight_matrix/Reshape_1Reshape*linear_to_mel_weight_matrix/split:output:14linear_to_mel_weight_matrix/Reshape_1/shape:output:0*
T0*
_output_shapes

:2|
+linear_to_mel_weight_matrix/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"   2   �
%linear_to_mel_weight_matrix/Reshape_2Reshape*linear_to_mel_weight_matrix/split:output:24linear_to_mel_weight_matrix/Reshape_2/shape:output:0*
T0*
_output_shapes

:2�
linear_to_mel_weight_matrix/subSub/linear_to_mel_weight_matrix/ExpandDims:output:0,linear_to_mel_weight_matrix/Reshape:output:0*
T0*
_output_shapes
:	�2�
!linear_to_mel_weight_matrix/sub_1Sub.linear_to_mel_weight_matrix/Reshape_1:output:0,linear_to_mel_weight_matrix/Reshape:output:0*
T0*
_output_shapes

:2�
%linear_to_mel_weight_matrix/truediv_1RealDiv#linear_to_mel_weight_matrix/sub:z:0%linear_to_mel_weight_matrix/sub_1:z:0*
T0*
_output_shapes
:	�2�
!linear_to_mel_weight_matrix/sub_2Sub.linear_to_mel_weight_matrix/Reshape_2:output:0/linear_to_mel_weight_matrix/ExpandDims:output:0*
T0*
_output_shapes
:	�2�
!linear_to_mel_weight_matrix/sub_3Sub.linear_to_mel_weight_matrix/Reshape_2:output:0.linear_to_mel_weight_matrix/Reshape_1:output:0*
T0*
_output_shapes

:2�
%linear_to_mel_weight_matrix/truediv_2RealDiv%linear_to_mel_weight_matrix/sub_2:z:0%linear_to_mel_weight_matrix/sub_3:z:0*
T0*
_output_shapes
:	�2�
#linear_to_mel_weight_matrix/MinimumMinimum)linear_to_mel_weight_matrix/truediv_1:z:0)linear_to_mel_weight_matrix/truediv_2:z:0*
T0*
_output_shapes
:	�2�
#linear_to_mel_weight_matrix/MaximumMaximum*linear_to_mel_weight_matrix/Const:output:0'linear_to_mel_weight_matrix/Minimum:z:0*
T0*
_output_shapes
:	�2�
$linear_to_mel_weight_matrix/paddingsConst*
_output_shapes

:*
dtype0*)
value B"               �
linear_to_mel_weight_matrixPad'linear_to_mel_weight_matrix/Maximum:z:0-linear_to_mel_weight_matrix/paddings:output:0*
T0*
_output_shapes
:	�2h
MatMulMatMulAbs:y:0$linear_to_mel_weight_matrix:output:0*
T0*
_output_shapes

:b2J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5W
addAddV2MatMul:product:0add/y:output:0*
T0*
_output_shapes

:b2<
LogLogadd:z:0*
T0*
_output_shapes

:b2Y
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������g

ExpandDims
ExpandDimsLog:y:0ExpandDims/dim:output:0*
T0*"
_output_shapes
:b2R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : {
ExpandDims_1
ExpandDimsExpandDims:output:0ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:b2�
StatefulPartitionedCallStatefulPartitionedCallExpandDims_1:output:0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37*3
Tin,
*2(*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*G
_read_only_resource_inputs)
'%	
 !"#$%&'*-
config_proto

CPU

GPU 2J 8� *3
f.R,
*__inference_signature_wrapper___call___753[
ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
���������{
ArgMaxArgMax StatefulPartitionedCall:output:0ArgMax/dimension:output:0*
T0*#
_output_shapes
:����������
GatherV2/paramsConst*
_output_shapes
:*
dtype0*[
valueRBPB
backgroundBdownBgoBleftBnoBoffBonBrightBstopBupByesBunknownO
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
GatherV2GatherV2GatherV2/params:output:0ArgMax:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:���������Z
IdentityIdentityArgMax:output:0^NoOp*
T0	*#
_output_shapes
:���������^

Identity_1IdentityGatherV2:output:0^NoOp*
T0*#
_output_shapes
:���������q

Identity_2Identity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesr
p: :b2:b2: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:$' 

_user_specified_name3036:$& 

_user_specified_name3034:$% 

_user_specified_name3032:$$ 

_user_specified_name3030:$# 

_user_specified_name3028:$" 

_user_specified_name3026:$! 

_user_specified_name3024:$  

_user_specified_name3022:$ 

_user_specified_name3020:$ 

_user_specified_name3018:$ 

_user_specified_name3016:$ 

_user_specified_name3014:$ 

_user_specified_name3012:$ 

_user_specified_name3010:$ 

_user_specified_name3008:$ 

_user_specified_name3006:$ 

_user_specified_name3004:$ 

_user_specified_name3002:$ 

_user_specified_name3000:$ 

_user_specified_name2998:$ 

_user_specified_name2996:$ 

_user_specified_name2994:$ 

_user_specified_name2992:$ 

_user_specified_name2990:$ 

_user_specified_name2988:$ 

_user_specified_name2986:$ 

_user_specified_name2984:$ 

_user_specified_name2982:$ 

_user_specified_name2980:$
 

_user_specified_name2978:$	 

_user_specified_name2976:$ 

_user_specified_name2974:$ 

_user_specified_name2972:$ 

_user_specified_name2970:$ 

_user_specified_name2968:$ 

_user_specified_name2966:$ 

_user_specified_name2964:LH
&
_output_shapes
:b2

_user_specified_name2962:LH
&
_output_shapes
:b2

_user_specified_name2960:= 9

_output_shapes
: 

_user_specified_nameinput
�"
�	
"__inference_signature_wrapper_2573	
input
unknown
	unknown_0#
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:$

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:$

unknown_19: 

unknown_20: 

unknown_21: 

unknown_22: 

unknown_23: 

unknown_24: $

unknown_25: P

unknown_26:P

unknown_27:P

unknown_28:P

unknown_29:P

unknown_30:P

unknown_31:
��

unknown_32:
��

unknown_33:	�

unknown_34:	�

unknown_35:

unknown_36:

unknown_37:
identity	

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37*3
Tin,
*2(*
Tout
2	*
_collective_manager_ids
 *E
_output_shapes3
1:���������:���������:���������*G
_read_only_resource_inputs)
'%	
 !"#$%&'*-
config_proto

CPU

GPU 2J 8� *"
fR
__inference___call___2397k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*#
_output_shapes
:���������m

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*#
_output_shapes
:���������q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:����������}:b2:b2: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:$' 

_user_specified_name2565:$& 

_user_specified_name2563:$% 

_user_specified_name2561:$$ 

_user_specified_name2559:$# 

_user_specified_name2557:$" 

_user_specified_name2555:$! 

_user_specified_name2553:$  

_user_specified_name2551:$ 

_user_specified_name2549:$ 

_user_specified_name2547:$ 

_user_specified_name2545:$ 

_user_specified_name2543:$ 

_user_specified_name2541:$ 

_user_specified_name2539:$ 

_user_specified_name2537:$ 

_user_specified_name2535:$ 

_user_specified_name2533:$ 

_user_specified_name2531:$ 

_user_specified_name2529:$ 

_user_specified_name2527:$ 

_user_specified_name2525:$ 

_user_specified_name2523:$ 

_user_specified_name2521:$ 

_user_specified_name2519:$ 

_user_specified_name2517:$ 

_user_specified_name2515:$ 

_user_specified_name2513:$ 

_user_specified_name2511:$ 

_user_specified_name2509:$
 

_user_specified_name2507:$	 

_user_specified_name2505:$ 

_user_specified_name2503:$ 

_user_specified_name2501:$ 

_user_specified_name2499:$ 

_user_specified_name2497:$ 

_user_specified_name2495:$ 

_user_specified_name2493:LH
&
_output_shapes
:b2

_user_specified_name2491:LH
&
_output_shapes
:b2

_user_specified_name2489:O K
(
_output_shapes
:����������}

_user_specified_nameinput
� 
�	
*__inference_signature_wrapper___call___753
imageinput_unnormalized
unknown
	unknown_0#
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:$

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:$

unknown_19: 

unknown_20: 

unknown_21: 

unknown_22: 

unknown_23: 

unknown_24: $

unknown_25: P

unknown_26:P

unknown_27:P

unknown_28:P

unknown_29:P

unknown_30:P

unknown_31:
��

unknown_32:
��

unknown_33:	�

unknown_34:	�

unknown_35:

unknown_36:

unknown_37:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallimageinput_unnormalizedunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37*3
Tin,
*2(*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*G
_read_only_resource_inputs)
'%	
 !"#$%&'*-
config_proto

CPU

GPU 2J 8� *!
fR
__inference___call___665<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������b2:b2:b2: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:$' 

_user_specified_name1213:$& 

_user_specified_name1211:$% 

_user_specified_name1209:$$ 

_user_specified_name1207:$# 

_user_specified_name1205:$" 

_user_specified_name1203:$! 

_user_specified_name1201:$  

_user_specified_name1199:$ 

_user_specified_name1197:$ 

_user_specified_name1195:$ 

_user_specified_name1193:$ 

_user_specified_name1191:$ 

_user_specified_name1189:$ 

_user_specified_name1187:$ 

_user_specified_name1185:$ 

_user_specified_name1183:$ 

_user_specified_name1181:$ 

_user_specified_name1179:$ 

_user_specified_name1177:$ 

_user_specified_name1175:$ 

_user_specified_name1173:$ 

_user_specified_name1171:$ 

_user_specified_name1169:$ 

_user_specified_name1167:$ 

_user_specified_name1165:$ 

_user_specified_name1163:$ 

_user_specified_name1161:$ 

_user_specified_name1159:$ 

_user_specified_name1157:$
 

_user_specified_name1155:$	 

_user_specified_name1153:$ 

_user_specified_name1151:$ 

_user_specified_name1149:$ 

_user_specified_name1147:$ 

_user_specified_name1145:$ 

_user_specified_name1143:$ 

_user_specified_name1141:LH
&
_output_shapes
:b2

_user_specified_name1139:LH
&
_output_shapes
:b2

_user_specified_name1137:h d
/
_output_shapes
:���������b2
1
_user_specified_nameimageinput_unnormalized
��
�	
__inference___call___2397	
input
unknown
	unknown_0#
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:$

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:$

unknown_19: 

unknown_20: 

unknown_21: 

unknown_22: 

unknown_23: 

unknown_24: $

unknown_25: P

unknown_26:P

unknown_27:P

unknown_28:P

unknown_29:P

unknown_30:P

unknown_31:
��

unknown_32:
��

unknown_33:	�

unknown_34:	�

unknown_35:

unknown_36:

unknown_37:
identity	

identity_1

identity_2��StatefulPartitionedCall<
SqueezeSqueezeinput*
T0*
_output_shapes
:H
ShapeShapeinput*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskH
sub/xConst*
_output_shapes
: *
dtype0*
value
B :�}S
subSubsub/x:output:0strided_slice:output:0*
T0*
_output_shapes
: K
	Maximum/xConst*
_output_shapes
: *
dtype0*
value	B : P
MaximumMaximumMaximum/x:output:0sub:z:0*
T0*
_output_shapes
: R
Pad/paddings/0/0Const*
_output_shapes
: *
dtype0*
value	B : l
Pad/paddings/0PackPad/paddings/0/0:output:0Maximum:z:0*
N*
T0*
_output_shapes
:_
Pad/paddingsPackPad/paddings/0:output:0*
N*
T0*
_output_shapes

:a
PadPadSqueeze:output:0Pad/paddings:output:0*
T0*#
_output_shapes
:���������Y
l2_normalize/SquareSquarePad:output:0*
T0*#
_output_shapes
:���������\
l2_normalize/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
l2_normalize/SumSuml2_normalize/Square:y:0l2_normalize/Const:output:0*
T0*
_output_shapes
:*
	keep_dims([
l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
l2_normalize/MaximumMaximuml2_normalize/Sum:output:0l2_normalize/Maximum/y:output:0*
T0*
_output_shapes
:Z
l2_normalize/RsqrtRsqrtl2_normalize/Maximum:z:0*
T0*
_output_shapes
:g
l2_normalizeMulPad:output:0l2_normalize/Rsqrt:y:0*
T0*#
_output_shapes
:���������T
stft/frame_lengthConst*
_output_shapes
: *
dtype0*
value
B :�R
stft/frame_stepConst*
_output_shapes
: *
dtype0*
value
B :�R
stft/fft_lengthConst*
_output_shapes
: *
dtype0*
value
B :�Z
stft/frame/axisConst*
_output_shapes
: *
dtype0*
valueB :
���������^
stft/frame/ShapeShapel2_normalize:z:0*
T0*
_output_shapes
::��Q
stft/frame/RankConst*
_output_shapes
: *
dtype0*
value	B :X
stft/frame/range/startConst*
_output_shapes
: *
dtype0*
value	B : X
stft/frame/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
stft/frame/rangeRangestft/frame/range/start:output:0stft/frame/Rank:output:0stft/frame/range/delta:output:0*
_output_shapes
:q
stft/frame/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������j
 stft/frame/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: j
 stft/frame/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
stft/frame/strided_sliceStridedSlicestft/frame/range:output:0'stft/frame/strided_slice/stack:output:0)stft/frame/strided_slice/stack_1:output:0)stft/frame/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskR
stft/frame/sub/yConst*
_output_shapes
: *
dtype0*
value	B :k
stft/frame/subSubstft/frame/Rank:output:0stft/frame/sub/y:output:0*
T0*
_output_shapes
: o
stft/frame/sub_1Substft/frame/sub:z:0!stft/frame/strided_slice:output:0*
T0*
_output_shapes
: U
stft/frame/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
stft/frame/packedPack!stft/frame/strided_slice:output:0stft/frame/packed/1:output:0stft/frame/sub_1:z:0*
N*
T0*
_output_shapes
:\
stft/frame/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : �
stft/frame/splitSplitVstft/frame/Shape:output:0stft/frame/packed:output:0#stft/frame/split/split_dim:output:0*

Tlen0*
T0*"
_output_shapes
: :: *
	num_split[
stft/frame/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB ]
stft/frame/Reshape/shape_1Const*
_output_shapes
: *
dtype0*
valueB ~
stft/frame/ReshapeReshapestft/frame/split:output:1#stft/frame/Reshape/shape_1:output:0*
T0*
_output_shapes
: Q
stft/frame/SizeConst*
_output_shapes
: *
dtype0*
value	B : S
stft/frame/Size_1Const*
_output_shapes
: *
dtype0*
value	B : R
stft/frame/ConstConst*
_output_shapes
: *
dtype0*
value	B : q
stft/frame/sub_2Substft/frame/Reshape:output:0stft/frame_length:output:0*
T0*
_output_shapes
: p
stft/frame/floordivFloorDivstft/frame/sub_2:z:0stft/frame_step:output:0*
T0*
_output_shapes
: R
stft/frame/add/xConst*
_output_shapes
: *
dtype0*
value	B :l
stft/frame/addAddV2stft/frame/add/x:output:0stft/frame/floordiv:z:0*
T0*
_output_shapes
: m
stft/frame/MaximumMaximumstft/frame/Const:output:0stft/frame/add:z:0*
T0*
_output_shapes
: V
stft/frame/gcd/ConstConst*
_output_shapes
: *
dtype0*
value	B :PY
stft/frame/floordiv_1/yConst*
_output_shapes
: *
dtype0*
value	B :P�
stft/frame/floordiv_1FloorDivstft/frame_length:output:0 stft/frame/floordiv_1/y:output:0*
T0*
_output_shapes
: Y
stft/frame/floordiv_2/yConst*
_output_shapes
: *
dtype0*
value	B :P~
stft/frame/floordiv_2FloorDivstft/frame_step:output:0 stft/frame/floordiv_2/y:output:0*
T0*
_output_shapes
: Y
stft/frame/floordiv_3/yConst*
_output_shapes
: *
dtype0*
value	B :P�
stft/frame/floordiv_3FloorDivstft/frame/Reshape:output:0 stft/frame/floordiv_3/y:output:0*
T0*
_output_shapes
: R
stft/frame/mul/yConst*
_output_shapes
: *
dtype0*
value	B :Pl
stft/frame/mulMulstft/frame/floordiv_3:z:0stft/frame/mul/y:output:0*
T0*
_output_shapes
: d
stft/frame/concat/values_1Packstft/frame/mul:z:0*
N*
T0*
_output_shapes
:X
stft/frame/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
stft/frame/concatConcatV2stft/frame/split:output:0#stft/frame/concat/values_1:output:0stft/frame/split:output:2stft/frame/concat/axis:output:0*
N*
T0*
_output_shapes
:`
stft/frame/concat_1/values_1/1Const*
_output_shapes
: *
dtype0*
value	B :P�
stft/frame/concat_1/values_1Packstft/frame/floordiv_3:z:0'stft/frame/concat_1/values_1/1:output:0*
N*
T0*
_output_shapes
:Z
stft/frame/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
stft/frame/concat_1ConcatV2stft/frame/split:output:0%stft/frame/concat_1/values_1:output:0stft/frame/split:output:2!stft/frame/concat_1/axis:output:0*
N*
T0*
_output_shapes
:_
stft/frame/zeros_likeConst*
_output_shapes
:*
dtype0*
valueB: t
*stft/frame/ones_like/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:\
stft/frame/ones_like/ConstConst*
_output_shapes
: *
dtype0*
value	B :�
stft/frame/ones_likeFill3stft/frame/ones_like/Shape/shape_as_tensor:output:0#stft/frame/ones_like/Const:output:0*
T0*
_output_shapes
:�
stft/frame/StridedSliceStridedSlicel2_normalize:z:0stft/frame/zeros_like:output:0stft/frame/concat:output:0stft/frame/ones_like:output:0*
Index0*
T0*#
_output_shapes
:����������
stft/frame/Reshape_1Reshape stft/frame/StridedSlice:output:0stft/frame/concat_1:output:0*
T0*'
_output_shapes
:���������PZ
stft/frame/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : Z
stft/frame/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
stft/frame/range_1Range!stft/frame/range_1/start:output:0stft/frame/Maximum:z:0!stft/frame/range_1/delta:output:0*#
_output_shapes
:���������}
stft/frame/mul_1Mulstft/frame/range_1:output:0stft/frame/floordiv_2:z:0*
T0*#
_output_shapes
:���������^
stft/frame/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :�
stft/frame/Reshape_2/shapePackstft/frame/Maximum:z:0%stft/frame/Reshape_2/shape/1:output:0*
N*
T0*
_output_shapes
:�
stft/frame/Reshape_2Reshapestft/frame/mul_1:z:0#stft/frame/Reshape_2/shape:output:0*
T0*'
_output_shapes
:���������Z
stft/frame/range_2/startConst*
_output_shapes
: *
dtype0*
value	B : Z
stft/frame/range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
stft/frame/range_2Range!stft/frame/range_2/start:output:0stft/frame/floordiv_1:z:0!stft/frame/range_2/delta:output:0*
_output_shapes
:^
stft/frame/Reshape_3/shape/0Const*
_output_shapes
: *
dtype0*
value	B :�
stft/frame/Reshape_3/shapePack%stft/frame/Reshape_3/shape/0:output:0stft/frame/floordiv_1:z:0*
N*
T0*
_output_shapes
:�
stft/frame/Reshape_3Reshapestft/frame/range_2:output:0#stft/frame/Reshape_3/shape:output:0*
T0*
_output_shapes

:�
stft/frame/add_1AddV2stft/frame/Reshape_2:output:0stft/frame/Reshape_3:output:0*
T0*'
_output_shapes
:���������}
stft/frame/packed_1Packstft/frame/Maximum:z:0stft/frame_length:output:0*
N*
T0*
_output_shapes
:�
stft/frame/GatherV2GatherV2stft/frame/Reshape_1:output:0stft/frame/add_1:z:0!stft/frame/strided_slice:output:0*
Taxis0*
Tindices0*
Tparams0*+
_output_shapes
:���������PZ
stft/frame/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
stft/frame/concat_2ConcatV2stft/frame/split:output:0stft/frame/packed_1:output:0stft/frame/split:output:2!stft/frame/concat_2/axis:output:0*
N*
T0*
_output_shapes
:�
stft/frame/Reshape_4Reshapestft/frame/GatherV2:output:0stft/frame/concat_2:output:0*
T0*(
_output_shapes
:����������[
stft/hann_window/periodicConst*
_output_shapes
: *
dtype0
*
value	B
 Zq
stft/hann_window/CastCast"stft/hann_window/periodic:output:0*

DstT0*

SrcT0
*
_output_shapes
: X
stft/hann_window/mod/yConst*
_output_shapes
: *
dtype0*
value	B :~
stft/hann_window/modFloorModstft/frame_length:output:0stft/hann_window/mod/y:output:0*
T0*
_output_shapes
: X
stft/hann_window/sub/xConst*
_output_shapes
: *
dtype0*
value	B :w
stft/hann_window/subSubstft/hann_window/sub/x:output:0stft/hann_window/mod:z:0*
T0*
_output_shapes
: q
stft/hann_window/mulMulstft/hann_window/Cast:y:0stft/hann_window/sub:z:0*
T0*
_output_shapes
: t
stft/hann_window/addAddV2stft/frame_length:output:0stft/hann_window/mul:z:0*
T0*
_output_shapes
: Z
stft/hann_window/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :{
stft/hann_window/sub_1Substft/hann_window/add:z:0!stft/hann_window/sub_1/y:output:0*
T0*
_output_shapes
: k
stft/hann_window/Cast_1Caststft/hann_window/sub_1:z:0*

DstT0*

SrcT0*
_output_shapes
: ^
stft/hann_window/range/startConst*
_output_shapes
: *
dtype0*
value	B : ^
stft/hann_window/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
stft/hann_window/rangeRange%stft/hann_window/range/start:output:0stft/frame_length:output:0%stft/hann_window/range/delta:output:0*
_output_shapes	
:�u
stft/hann_window/Cast_2Caststft/hann_window/range:output:0*

DstT0*

SrcT0*
_output_shapes	
:�[
stft/hann_window/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *��@�
stft/hann_window/mul_1Mulstft/hann_window/Const:output:0stft/hann_window/Cast_2:y:0*
T0*
_output_shapes	
:��
stft/hann_window/truedivRealDivstft/hann_window/mul_1:z:0stft/hann_window/Cast_1:y:0*
T0*
_output_shapes	
:�_
stft/hann_window/CosCosstft/hann_window/truediv:z:0*
T0*
_output_shapes	
:�]
stft/hann_window/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
stft/hann_window/mul_2Mul!stft/hann_window/mul_2/x:output:0stft/hann_window/Cos:y:0*
T0*
_output_shapes	
:�]
stft/hann_window/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
stft/hann_window/sub_2Sub!stft/hann_window/sub_2/x:output:0stft/hann_window/mul_2:z:0*
T0*
_output_shapes	
:�}
stft/mulMulstft/frame/Reshape_4:output:0stft/hann_window/sub_2:z:0*
T0*(
_output_shapes
:����������`
stft/rfft/packedPackstft/fft_length:output:0*
N*
T0*
_output_shapes
:w
stft/rfft/Pad/paddingsConst*
_output_shapes

:*
dtype0*)
value B"            p   v
stft/rfft/PadPadstft/mul:z:0stft/rfft/Pad/paddings:output:0*
T0*(
_output_shapes
:����������_
stft/rfft/fft_lengthConst*
_output_shapes
:*
dtype0*
valueB:�r
	stft/rfftRFFTstft/rfft/Pad:output:0stft/rfft/fft_length:output:0*(
_output_shapes
:����������O
Abs
ComplexAbsstft/rfft:output:0*(
_output_shapes
:����������L
Shape_1ShapeAbs:y:0*
T0*
_output_shapes
::��h
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
)linear_to_mel_weight_matrix/sample_rate/xConst*
_output_shapes
: *
dtype0*
value
B :�}�
'linear_to_mel_weight_matrix/sample_rateCast2linear_to_mel_weight_matrix/sample_rate/x:output:0*

DstT0*

SrcT0*
_output_shapes
: q
,linear_to_mel_weight_matrix/lower_edge_hertzConst*
_output_shapes
: *
dtype0*
valueB
 *  �Bq
,linear_to_mel_weight_matrix/upper_edge_hertzConst*
_output_shapes
: *
dtype0*
valueB
 * �mEf
!linear_to_mel_weight_matrix/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    j
%linear_to_mel_weight_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
#linear_to_mel_weight_matrix/truedivRealDiv+linear_to_mel_weight_matrix/sample_rate:y:0.linear_to_mel_weight_matrix/truediv/y:output:0*
T0*
_output_shapes
: {
)linear_to_mel_weight_matrix/linspace/CastCaststrided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: �
+linear_to_mel_weight_matrix/linspace/Cast_1Cast-linear_to_mel_weight_matrix/linspace/Cast:y:0*

DstT0*

SrcT0*
_output_shapes
: m
*linear_to_mel_weight_matrix/linspace/ShapeConst*
_output_shapes
: *
dtype0*
valueB o
,linear_to_mel_weight_matrix/linspace/Shape_1Const*
_output_shapes
: *
dtype0*
valueB �
2linear_to_mel_weight_matrix/linspace/BroadcastArgsBroadcastArgs3linear_to_mel_weight_matrix/linspace/Shape:output:05linear_to_mel_weight_matrix/linspace/Shape_1:output:0*
_output_shapes
: �
0linear_to_mel_weight_matrix/linspace/BroadcastToBroadcastTo*linear_to_mel_weight_matrix/Const:output:07linear_to_mel_weight_matrix/linspace/BroadcastArgs:r0:0*
T0*
_output_shapes
: �
2linear_to_mel_weight_matrix/linspace/BroadcastTo_1BroadcastTo'linear_to_mel_weight_matrix/truediv:z:07linear_to_mel_weight_matrix/linspace/BroadcastArgs:r0:0*
T0*
_output_shapes
: u
3linear_to_mel_weight_matrix/linspace/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : �
/linear_to_mel_weight_matrix/linspace/ExpandDims
ExpandDims9linear_to_mel_weight_matrix/linspace/BroadcastTo:output:0<linear_to_mel_weight_matrix/linspace/ExpandDims/dim:output:0*
T0*
_output_shapes
:w
5linear_to_mel_weight_matrix/linspace/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
1linear_to_mel_weight_matrix/linspace/ExpandDims_1
ExpandDims;linear_to_mel_weight_matrix/linspace/BroadcastTo_1:output:0>linear_to_mel_weight_matrix/linspace/ExpandDims_1/dim:output:0*
T0*
_output_shapes
:v
,linear_to_mel_weight_matrix/linspace/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:v
,linear_to_mel_weight_matrix/linspace/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:�
8linear_to_mel_weight_matrix/linspace/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
:linear_to_mel_weight_matrix/linspace/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
:linear_to_mel_weight_matrix/linspace/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
2linear_to_mel_weight_matrix/linspace/strided_sliceStridedSlice5linear_to_mel_weight_matrix/linspace/Shape_3:output:0Alinear_to_mel_weight_matrix/linspace/strided_slice/stack:output:0Clinear_to_mel_weight_matrix/linspace/strided_slice/stack_1:output:0Clinear_to_mel_weight_matrix/linspace/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
*linear_to_mel_weight_matrix/linspace/add/yConst*
_output_shapes
: *
dtype0*
value	B : �
(linear_to_mel_weight_matrix/linspace/addAddV2;linear_to_mel_weight_matrix/linspace/strided_slice:output:03linear_to_mel_weight_matrix/linspace/add/y:output:0*
T0*
_output_shapes
: y
7linear_to_mel_weight_matrix/linspace/SelectV2/conditionConst*
_output_shapes
: *
dtype0
*
value	B
 Zq
/linear_to_mel_weight_matrix/linspace/SelectV2/tConst*
_output_shapes
: *
dtype0*
value	B : �
-linear_to_mel_weight_matrix/linspace/SelectV2SelectV2@linear_to_mel_weight_matrix/linspace/SelectV2/condition:output:08linear_to_mel_weight_matrix/linspace/SelectV2/t:output:0,linear_to_mel_weight_matrix/linspace/add:z:0*
T0*
_output_shapes
: l
*linear_to_mel_weight_matrix/linspace/sub/yConst*
_output_shapes
: *
dtype0*
value	B :�
(linear_to_mel_weight_matrix/linspace/subSub-linear_to_mel_weight_matrix/linspace/Cast:y:03linear_to_mel_weight_matrix/linspace/sub/y:output:0*
T0*
_output_shapes
: p
.linear_to_mel_weight_matrix/linspace/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B : �
,linear_to_mel_weight_matrix/linspace/MaximumMaximum,linear_to_mel_weight_matrix/linspace/sub:z:07linear_to_mel_weight_matrix/linspace/Maximum/y:output:0*
T0*
_output_shapes
: n
,linear_to_mel_weight_matrix/linspace/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
*linear_to_mel_weight_matrix/linspace/sub_1Sub-linear_to_mel_weight_matrix/linspace/Cast:y:05linear_to_mel_weight_matrix/linspace/sub_1/y:output:0*
T0*
_output_shapes
: r
0linear_to_mel_weight_matrix/linspace/Maximum_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
.linear_to_mel_weight_matrix/linspace/Maximum_1Maximum.linear_to_mel_weight_matrix/linspace/sub_1:z:09linear_to_mel_weight_matrix/linspace/Maximum_1/y:output:0*
T0*
_output_shapes
: �
*linear_to_mel_weight_matrix/linspace/sub_2Sub:linear_to_mel_weight_matrix/linspace/ExpandDims_1:output:08linear_to_mel_weight_matrix/linspace/ExpandDims:output:0*
T0*
_output_shapes
:�
+linear_to_mel_weight_matrix/linspace/Cast_2Cast2linear_to_mel_weight_matrix/linspace/Maximum_1:z:0*

DstT0*

SrcT0*
_output_shapes
: �
,linear_to_mel_weight_matrix/linspace/truedivRealDiv.linear_to_mel_weight_matrix/linspace/sub_2:z:0/linear_to_mel_weight_matrix/linspace/Cast_2:y:0*
T0*
_output_shapes
:u
3linear_to_mel_weight_matrix/linspace/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : �
1linear_to_mel_weight_matrix/linspace/GreaterEqualGreaterEqual-linear_to_mel_weight_matrix/linspace/Cast:y:0<linear_to_mel_weight_matrix/linspace/GreaterEqual/y:output:0*
T0*
_output_shapes
: |
1linear_to_mel_weight_matrix/linspace/SelectV2_1/eConst*
_output_shapes
: *
dtype0*
valueB :
����������
/linear_to_mel_weight_matrix/linspace/SelectV2_1SelectV25linear_to_mel_weight_matrix/linspace/GreaterEqual:z:02linear_to_mel_weight_matrix/linspace/Maximum_1:z:0:linear_to_mel_weight_matrix/linspace/SelectV2_1/e:output:0*
T0*
_output_shapes
: r
0linear_to_mel_weight_matrix/linspace/range/startConst*
_output_shapes
: *
dtype0	*
value	B	 Rr
0linear_to_mel_weight_matrix/linspace/range/deltaConst*
_output_shapes
: *
dtype0	*
value	B	 R�
/linear_to_mel_weight_matrix/linspace/range/CastCast8linear_to_mel_weight_matrix/linspace/SelectV2_1:output:0*

DstT0	*

SrcT0*
_output_shapes
: �
*linear_to_mel_weight_matrix/linspace/rangeRange9linear_to_mel_weight_matrix/linspace/range/start:output:03linear_to_mel_weight_matrix/linspace/range/Cast:y:09linear_to_mel_weight_matrix/linspace/range/delta:output:0*

Tidx0	*
_output_shapes	
:��
+linear_to_mel_weight_matrix/linspace/Cast_3Cast3linear_to_mel_weight_matrix/linspace/range:output:0*

DstT0*

SrcT0	*
_output_shapes	
:�t
2linear_to_mel_weight_matrix/linspace/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : t
2linear_to_mel_weight_matrix/linspace/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
,linear_to_mel_weight_matrix/linspace/range_1Range;linear_to_mel_weight_matrix/linspace/range_1/start:output:0;linear_to_mel_weight_matrix/linspace/strided_slice:output:0;linear_to_mel_weight_matrix/linspace/range_1/delta:output:0*
_output_shapes
:�
*linear_to_mel_weight_matrix/linspace/EqualEqual6linear_to_mel_weight_matrix/linspace/SelectV2:output:05linear_to_mel_weight_matrix/linspace/range_1:output:0*
T0*
_output_shapes
:s
1linear_to_mel_weight_matrix/linspace/SelectV2_2/eConst*
_output_shapes
: *
dtype0*
value	B :�
/linear_to_mel_weight_matrix/linspace/SelectV2_2SelectV2.linear_to_mel_weight_matrix/linspace/Equal:z:00linear_to_mel_weight_matrix/linspace/Maximum:z:0:linear_to_mel_weight_matrix/linspace/SelectV2_2/e:output:0*
T0*
_output_shapes
:�
,linear_to_mel_weight_matrix/linspace/ReshapeReshape/linear_to_mel_weight_matrix/linspace/Cast_3:y:08linear_to_mel_weight_matrix/linspace/SelectV2_2:output:0*
T0*
_output_shapes	
:��
(linear_to_mel_weight_matrix/linspace/mulMul0linear_to_mel_weight_matrix/linspace/truediv:z:05linear_to_mel_weight_matrix/linspace/Reshape:output:0*
T0*
_output_shapes	
:��
*linear_to_mel_weight_matrix/linspace/add_1AddV28linear_to_mel_weight_matrix/linspace/ExpandDims:output:0,linear_to_mel_weight_matrix/linspace/mul:z:0*
T0*
_output_shapes	
:��
+linear_to_mel_weight_matrix/linspace/concatConcatV28linear_to_mel_weight_matrix/linspace/ExpandDims:output:0.linear_to_mel_weight_matrix/linspace/add_1:z:0:linear_to_mel_weight_matrix/linspace/ExpandDims_1:output:06linear_to_mel_weight_matrix/linspace/SelectV2:output:0*
N*
T0*
_output_shapes	
:�y
/linear_to_mel_weight_matrix/linspace/zeros_likeConst*
_output_shapes
:*
dtype0*
valueB: l
*linear_to_mel_weight_matrix/linspace/ConstConst*
_output_shapes
: *
dtype0*
value	B :�
:linear_to_mel_weight_matrix/linspace/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
<linear_to_mel_weight_matrix/linspace/strided_slice_1/stack_1Pack6linear_to_mel_weight_matrix/linspace/SelectV2:output:0*
N*
T0*
_output_shapes
:�
<linear_to_mel_weight_matrix/linspace/strided_slice_1/stack_2Pack3linear_to_mel_weight_matrix/linspace/Const:output:0*
N*
T0*
_output_shapes
:�
4linear_to_mel_weight_matrix/linspace/strided_slice_1StridedSlice5linear_to_mel_weight_matrix/linspace/Shape_2:output:0Clinear_to_mel_weight_matrix/linspace/strided_slice_1/stack:output:0Elinear_to_mel_weight_matrix/linspace/strided_slice_1/stack_1:output:0Elinear_to_mel_weight_matrix/linspace/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: ~
4linear_to_mel_weight_matrix/linspace/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:�
.linear_to_mel_weight_matrix/linspace/Reshape_1Reshape-linear_to_mel_weight_matrix/linspace/Cast:y:0=linear_to_mel_weight_matrix/linspace/Reshape_1/shape:output:0*
T0*
_output_shapes
:n
,linear_to_mel_weight_matrix/linspace/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :�
*linear_to_mel_weight_matrix/linspace/add_2AddV26linear_to_mel_weight_matrix/linspace/SelectV2:output:05linear_to_mel_weight_matrix/linspace/add_2/y:output:0*
T0*
_output_shapes
: n
,linear_to_mel_weight_matrix/linspace/Const_1Const*
_output_shapes
: *
dtype0*
value	B : n
,linear_to_mel_weight_matrix/linspace/Const_2Const*
_output_shapes
: *
dtype0*
value	B :�
:linear_to_mel_weight_matrix/linspace/strided_slice_2/stackPack.linear_to_mel_weight_matrix/linspace/add_2:z:0*
N*
T0*
_output_shapes
:�
<linear_to_mel_weight_matrix/linspace/strided_slice_2/stack_1Pack5linear_to_mel_weight_matrix/linspace/Const_1:output:0*
N*
T0*
_output_shapes
:�
<linear_to_mel_weight_matrix/linspace/strided_slice_2/stack_2Pack5linear_to_mel_weight_matrix/linspace/Const_2:output:0*
N*
T0*
_output_shapes
:�
4linear_to_mel_weight_matrix/linspace/strided_slice_2StridedSlice5linear_to_mel_weight_matrix/linspace/Shape_2:output:0Clinear_to_mel_weight_matrix/linspace/strided_slice_2/stack:output:0Elinear_to_mel_weight_matrix/linspace/strided_slice_2/stack_1:output:0Elinear_to_mel_weight_matrix/linspace/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_maskt
2linear_to_mel_weight_matrix/linspace/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
-linear_to_mel_weight_matrix/linspace/concat_1ConcatV2=linear_to_mel_weight_matrix/linspace/strided_slice_1:output:07linear_to_mel_weight_matrix/linspace/Reshape_1:output:0=linear_to_mel_weight_matrix/linspace/strided_slice_2:output:0;linear_to_mel_weight_matrix/linspace/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
*linear_to_mel_weight_matrix/linspace/SliceSlice4linear_to_mel_weight_matrix/linspace/concat:output:08linear_to_mel_weight_matrix/linspace/zeros_like:output:06linear_to_mel_weight_matrix/linspace/concat_1:output:0*
Index0*
T0*
_output_shapes	
:�y
/linear_to_mel_weight_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:{
1linear_to_mel_weight_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: {
1linear_to_mel_weight_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
)linear_to_mel_weight_matrix/strided_sliceStridedSlice3linear_to_mel_weight_matrix/linspace/Slice:output:08linear_to_mel_weight_matrix/strided_slice/stack:output:0:linear_to_mel_weight_matrix/strided_slice/stack_1:output:0:linear_to_mel_weight_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes	
:�*
end_maskw
2linear_to_mel_weight_matrix/hertz_to_mel/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  /D�
0linear_to_mel_weight_matrix/hertz_to_mel/truedivRealDiv2linear_to_mel_weight_matrix/strided_slice:output:0;linear_to_mel_weight_matrix/hertz_to_mel/truediv/y:output:0*
T0*
_output_shapes	
:�s
.linear_to_mel_weight_matrix/hertz_to_mel/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
,linear_to_mel_weight_matrix/hertz_to_mel/addAddV27linear_to_mel_weight_matrix/hertz_to_mel/add/x:output:04linear_to_mel_weight_matrix/hertz_to_mel/truediv:z:0*
T0*
_output_shapes	
:��
,linear_to_mel_weight_matrix/hertz_to_mel/LogLog0linear_to_mel_weight_matrix/hertz_to_mel/add:z:0*
T0*
_output_shapes	
:�s
.linear_to_mel_weight_matrix/hertz_to_mel/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 * ��D�
,linear_to_mel_weight_matrix/hertz_to_mel/mulMul7linear_to_mel_weight_matrix/hertz_to_mel/mul/x:output:00linear_to_mel_weight_matrix/hertz_to_mel/Log:y:0*
T0*
_output_shapes	
:�l
*linear_to_mel_weight_matrix/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
&linear_to_mel_weight_matrix/ExpandDims
ExpandDims0linear_to_mel_weight_matrix/hertz_to_mel/mul:z:03linear_to_mel_weight_matrix/ExpandDims/dim:output:0*
T0*
_output_shapes
:	�y
4linear_to_mel_weight_matrix/hertz_to_mel_1/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  /D�
2linear_to_mel_weight_matrix/hertz_to_mel_1/truedivRealDiv5linear_to_mel_weight_matrix/lower_edge_hertz:output:0=linear_to_mel_weight_matrix/hertz_to_mel_1/truediv/y:output:0*
T0*
_output_shapes
: u
0linear_to_mel_weight_matrix/hertz_to_mel_1/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
.linear_to_mel_weight_matrix/hertz_to_mel_1/addAddV29linear_to_mel_weight_matrix/hertz_to_mel_1/add/x:output:06linear_to_mel_weight_matrix/hertz_to_mel_1/truediv:z:0*
T0*
_output_shapes
: �
.linear_to_mel_weight_matrix/hertz_to_mel_1/LogLog2linear_to_mel_weight_matrix/hertz_to_mel_1/add:z:0*
T0*
_output_shapes
: u
0linear_to_mel_weight_matrix/hertz_to_mel_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 * ��D�
.linear_to_mel_weight_matrix/hertz_to_mel_1/mulMul9linear_to_mel_weight_matrix/hertz_to_mel_1/mul/x:output:02linear_to_mel_weight_matrix/hertz_to_mel_1/Log:y:0*
T0*
_output_shapes
: y
4linear_to_mel_weight_matrix/hertz_to_mel_2/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  /D�
2linear_to_mel_weight_matrix/hertz_to_mel_2/truedivRealDiv5linear_to_mel_weight_matrix/upper_edge_hertz:output:0=linear_to_mel_weight_matrix/hertz_to_mel_2/truediv/y:output:0*
T0*
_output_shapes
: u
0linear_to_mel_weight_matrix/hertz_to_mel_2/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
.linear_to_mel_weight_matrix/hertz_to_mel_2/addAddV29linear_to_mel_weight_matrix/hertz_to_mel_2/add/x:output:06linear_to_mel_weight_matrix/hertz_to_mel_2/truediv:z:0*
T0*
_output_shapes
: �
.linear_to_mel_weight_matrix/hertz_to_mel_2/LogLog2linear_to_mel_weight_matrix/hertz_to_mel_2/add:z:0*
T0*
_output_shapes
: u
0linear_to_mel_weight_matrix/hertz_to_mel_2/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 * ��D�
.linear_to_mel_weight_matrix/hertz_to_mel_2/mulMul9linear_to_mel_weight_matrix/hertz_to_mel_2/mul/x:output:02linear_to_mel_weight_matrix/hertz_to_mel_2/Log:y:0*
T0*
_output_shapes
: l
*linear_to_mel_weight_matrix/linspace_1/numConst*
_output_shapes
: *
dtype0*
value	B :4�
+linear_to_mel_weight_matrix/linspace_1/CastCast3linear_to_mel_weight_matrix/linspace_1/num:output:0*

DstT0*

SrcT0*
_output_shapes
: �
-linear_to_mel_weight_matrix/linspace_1/Cast_1Cast/linear_to_mel_weight_matrix/linspace_1/Cast:y:0*

DstT0*

SrcT0*
_output_shapes
: o
,linear_to_mel_weight_matrix/linspace_1/ShapeConst*
_output_shapes
: *
dtype0*
valueB q
.linear_to_mel_weight_matrix/linspace_1/Shape_1Const*
_output_shapes
: *
dtype0*
valueB �
4linear_to_mel_weight_matrix/linspace_1/BroadcastArgsBroadcastArgs5linear_to_mel_weight_matrix/linspace_1/Shape:output:07linear_to_mel_weight_matrix/linspace_1/Shape_1:output:0*
_output_shapes
: �
2linear_to_mel_weight_matrix/linspace_1/BroadcastToBroadcastTo2linear_to_mel_weight_matrix/hertz_to_mel_1/mul:z:09linear_to_mel_weight_matrix/linspace_1/BroadcastArgs:r0:0*
T0*
_output_shapes
: �
4linear_to_mel_weight_matrix/linspace_1/BroadcastTo_1BroadcastTo2linear_to_mel_weight_matrix/hertz_to_mel_2/mul:z:09linear_to_mel_weight_matrix/linspace_1/BroadcastArgs:r0:0*
T0*
_output_shapes
: w
5linear_to_mel_weight_matrix/linspace_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : �
1linear_to_mel_weight_matrix/linspace_1/ExpandDims
ExpandDims;linear_to_mel_weight_matrix/linspace_1/BroadcastTo:output:0>linear_to_mel_weight_matrix/linspace_1/ExpandDims/dim:output:0*
T0*
_output_shapes
:y
7linear_to_mel_weight_matrix/linspace_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
3linear_to_mel_weight_matrix/linspace_1/ExpandDims_1
ExpandDims=linear_to_mel_weight_matrix/linspace_1/BroadcastTo_1:output:0@linear_to_mel_weight_matrix/linspace_1/ExpandDims_1/dim:output:0*
T0*
_output_shapes
:x
.linear_to_mel_weight_matrix/linspace_1/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:x
.linear_to_mel_weight_matrix/linspace_1/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:�
:linear_to_mel_weight_matrix/linspace_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
<linear_to_mel_weight_matrix/linspace_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
<linear_to_mel_weight_matrix/linspace_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
4linear_to_mel_weight_matrix/linspace_1/strided_sliceStridedSlice7linear_to_mel_weight_matrix/linspace_1/Shape_3:output:0Clinear_to_mel_weight_matrix/linspace_1/strided_slice/stack:output:0Elinear_to_mel_weight_matrix/linspace_1/strided_slice/stack_1:output:0Elinear_to_mel_weight_matrix/linspace_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
,linear_to_mel_weight_matrix/linspace_1/add/yConst*
_output_shapes
: *
dtype0*
value	B : �
*linear_to_mel_weight_matrix/linspace_1/addAddV2=linear_to_mel_weight_matrix/linspace_1/strided_slice:output:05linear_to_mel_weight_matrix/linspace_1/add/y:output:0*
T0*
_output_shapes
: {
9linear_to_mel_weight_matrix/linspace_1/SelectV2/conditionConst*
_output_shapes
: *
dtype0
*
value	B
 Zs
1linear_to_mel_weight_matrix/linspace_1/SelectV2/tConst*
_output_shapes
: *
dtype0*
value	B : �
/linear_to_mel_weight_matrix/linspace_1/SelectV2SelectV2Blinear_to_mel_weight_matrix/linspace_1/SelectV2/condition:output:0:linear_to_mel_weight_matrix/linspace_1/SelectV2/t:output:0.linear_to_mel_weight_matrix/linspace_1/add:z:0*
T0*
_output_shapes
: n
,linear_to_mel_weight_matrix/linspace_1/sub/yConst*
_output_shapes
: *
dtype0*
value	B :�
*linear_to_mel_weight_matrix/linspace_1/subSub/linear_to_mel_weight_matrix/linspace_1/Cast:y:05linear_to_mel_weight_matrix/linspace_1/sub/y:output:0*
T0*
_output_shapes
: r
0linear_to_mel_weight_matrix/linspace_1/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B : �
.linear_to_mel_weight_matrix/linspace_1/MaximumMaximum.linear_to_mel_weight_matrix/linspace_1/sub:z:09linear_to_mel_weight_matrix/linspace_1/Maximum/y:output:0*
T0*
_output_shapes
: p
.linear_to_mel_weight_matrix/linspace_1/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
,linear_to_mel_weight_matrix/linspace_1/sub_1Sub/linear_to_mel_weight_matrix/linspace_1/Cast:y:07linear_to_mel_weight_matrix/linspace_1/sub_1/y:output:0*
T0*
_output_shapes
: t
2linear_to_mel_weight_matrix/linspace_1/Maximum_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
0linear_to_mel_weight_matrix/linspace_1/Maximum_1Maximum0linear_to_mel_weight_matrix/linspace_1/sub_1:z:0;linear_to_mel_weight_matrix/linspace_1/Maximum_1/y:output:0*
T0*
_output_shapes
: �
,linear_to_mel_weight_matrix/linspace_1/sub_2Sub<linear_to_mel_weight_matrix/linspace_1/ExpandDims_1:output:0:linear_to_mel_weight_matrix/linspace_1/ExpandDims:output:0*
T0*
_output_shapes
:�
-linear_to_mel_weight_matrix/linspace_1/Cast_2Cast4linear_to_mel_weight_matrix/linspace_1/Maximum_1:z:0*

DstT0*

SrcT0*
_output_shapes
: �
.linear_to_mel_weight_matrix/linspace_1/truedivRealDiv0linear_to_mel_weight_matrix/linspace_1/sub_2:z:01linear_to_mel_weight_matrix/linspace_1/Cast_2:y:0*
T0*
_output_shapes
:w
5linear_to_mel_weight_matrix/linspace_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : �
3linear_to_mel_weight_matrix/linspace_1/GreaterEqualGreaterEqual/linear_to_mel_weight_matrix/linspace_1/Cast:y:0>linear_to_mel_weight_matrix/linspace_1/GreaterEqual/y:output:0*
T0*
_output_shapes
: ~
3linear_to_mel_weight_matrix/linspace_1/SelectV2_1/eConst*
_output_shapes
: *
dtype0*
valueB :
����������
1linear_to_mel_weight_matrix/linspace_1/SelectV2_1SelectV27linear_to_mel_weight_matrix/linspace_1/GreaterEqual:z:04linear_to_mel_weight_matrix/linspace_1/Maximum_1:z:0<linear_to_mel_weight_matrix/linspace_1/SelectV2_1/e:output:0*
T0*
_output_shapes
: t
2linear_to_mel_weight_matrix/linspace_1/range/startConst*
_output_shapes
: *
dtype0	*
value	B	 Rt
2linear_to_mel_weight_matrix/linspace_1/range/deltaConst*
_output_shapes
: *
dtype0	*
value	B	 R�
1linear_to_mel_weight_matrix/linspace_1/range/CastCast:linear_to_mel_weight_matrix/linspace_1/SelectV2_1:output:0*

DstT0	*

SrcT0*
_output_shapes
: �
,linear_to_mel_weight_matrix/linspace_1/rangeRange;linear_to_mel_weight_matrix/linspace_1/range/start:output:05linear_to_mel_weight_matrix/linspace_1/range/Cast:y:0;linear_to_mel_weight_matrix/linspace_1/range/delta:output:0*

Tidx0	*
_output_shapes
:2�
-linear_to_mel_weight_matrix/linspace_1/Cast_3Cast5linear_to_mel_weight_matrix/linspace_1/range:output:0*

DstT0*

SrcT0	*
_output_shapes
:2v
4linear_to_mel_weight_matrix/linspace_1/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : v
4linear_to_mel_weight_matrix/linspace_1/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
.linear_to_mel_weight_matrix/linspace_1/range_1Range=linear_to_mel_weight_matrix/linspace_1/range_1/start:output:0=linear_to_mel_weight_matrix/linspace_1/strided_slice:output:0=linear_to_mel_weight_matrix/linspace_1/range_1/delta:output:0*
_output_shapes
:�
,linear_to_mel_weight_matrix/linspace_1/EqualEqual8linear_to_mel_weight_matrix/linspace_1/SelectV2:output:07linear_to_mel_weight_matrix/linspace_1/range_1:output:0*
T0*
_output_shapes
:u
3linear_to_mel_weight_matrix/linspace_1/SelectV2_2/eConst*
_output_shapes
: *
dtype0*
value	B :�
1linear_to_mel_weight_matrix/linspace_1/SelectV2_2SelectV20linear_to_mel_weight_matrix/linspace_1/Equal:z:02linear_to_mel_weight_matrix/linspace_1/Maximum:z:0<linear_to_mel_weight_matrix/linspace_1/SelectV2_2/e:output:0*
T0*
_output_shapes
:�
.linear_to_mel_weight_matrix/linspace_1/ReshapeReshape1linear_to_mel_weight_matrix/linspace_1/Cast_3:y:0:linear_to_mel_weight_matrix/linspace_1/SelectV2_2:output:0*
T0*
_output_shapes
:2�
*linear_to_mel_weight_matrix/linspace_1/mulMul2linear_to_mel_weight_matrix/linspace_1/truediv:z:07linear_to_mel_weight_matrix/linspace_1/Reshape:output:0*
T0*
_output_shapes
:2�
,linear_to_mel_weight_matrix/linspace_1/add_1AddV2:linear_to_mel_weight_matrix/linspace_1/ExpandDims:output:0.linear_to_mel_weight_matrix/linspace_1/mul:z:0*
T0*
_output_shapes
:2�
-linear_to_mel_weight_matrix/linspace_1/concatConcatV2:linear_to_mel_weight_matrix/linspace_1/ExpandDims:output:00linear_to_mel_weight_matrix/linspace_1/add_1:z:0<linear_to_mel_weight_matrix/linspace_1/ExpandDims_1:output:08linear_to_mel_weight_matrix/linspace_1/SelectV2:output:0*
N*
T0*
_output_shapes
:4{
1linear_to_mel_weight_matrix/linspace_1/zeros_likeConst*
_output_shapes
:*
dtype0*
valueB: n
,linear_to_mel_weight_matrix/linspace_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :�
<linear_to_mel_weight_matrix/linspace_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
>linear_to_mel_weight_matrix/linspace_1/strided_slice_1/stack_1Pack8linear_to_mel_weight_matrix/linspace_1/SelectV2:output:0*
N*
T0*
_output_shapes
:�
>linear_to_mel_weight_matrix/linspace_1/strided_slice_1/stack_2Pack5linear_to_mel_weight_matrix/linspace_1/Const:output:0*
N*
T0*
_output_shapes
:�
6linear_to_mel_weight_matrix/linspace_1/strided_slice_1StridedSlice7linear_to_mel_weight_matrix/linspace_1/Shape_2:output:0Elinear_to_mel_weight_matrix/linspace_1/strided_slice_1/stack:output:0Glinear_to_mel_weight_matrix/linspace_1/strided_slice_1/stack_1:output:0Glinear_to_mel_weight_matrix/linspace_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: �
6linear_to_mel_weight_matrix/linspace_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:�
0linear_to_mel_weight_matrix/linspace_1/Reshape_1Reshape/linear_to_mel_weight_matrix/linspace_1/Cast:y:0?linear_to_mel_weight_matrix/linspace_1/Reshape_1/shape:output:0*
T0*
_output_shapes
:p
.linear_to_mel_weight_matrix/linspace_1/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :�
,linear_to_mel_weight_matrix/linspace_1/add_2AddV28linear_to_mel_weight_matrix/linspace_1/SelectV2:output:07linear_to_mel_weight_matrix/linspace_1/add_2/y:output:0*
T0*
_output_shapes
: p
.linear_to_mel_weight_matrix/linspace_1/Const_1Const*
_output_shapes
: *
dtype0*
value	B : p
.linear_to_mel_weight_matrix/linspace_1/Const_2Const*
_output_shapes
: *
dtype0*
value	B :�
<linear_to_mel_weight_matrix/linspace_1/strided_slice_2/stackPack0linear_to_mel_weight_matrix/linspace_1/add_2:z:0*
N*
T0*
_output_shapes
:�
>linear_to_mel_weight_matrix/linspace_1/strided_slice_2/stack_1Pack7linear_to_mel_weight_matrix/linspace_1/Const_1:output:0*
N*
T0*
_output_shapes
:�
>linear_to_mel_weight_matrix/linspace_1/strided_slice_2/stack_2Pack7linear_to_mel_weight_matrix/linspace_1/Const_2:output:0*
N*
T0*
_output_shapes
:�
6linear_to_mel_weight_matrix/linspace_1/strided_slice_2StridedSlice7linear_to_mel_weight_matrix/linspace_1/Shape_2:output:0Elinear_to_mel_weight_matrix/linspace_1/strided_slice_2/stack:output:0Glinear_to_mel_weight_matrix/linspace_1/strided_slice_2/stack_1:output:0Glinear_to_mel_weight_matrix/linspace_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_maskv
4linear_to_mel_weight_matrix/linspace_1/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
/linear_to_mel_weight_matrix/linspace_1/concat_1ConcatV2?linear_to_mel_weight_matrix/linspace_1/strided_slice_1:output:09linear_to_mel_weight_matrix/linspace_1/Reshape_1:output:0?linear_to_mel_weight_matrix/linspace_1/strided_slice_2:output:0=linear_to_mel_weight_matrix/linspace_1/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
,linear_to_mel_weight_matrix/linspace_1/SliceSlice6linear_to_mel_weight_matrix/linspace_1/concat:output:0:linear_to_mel_weight_matrix/linspace_1/zeros_like:output:08linear_to_mel_weight_matrix/linspace_1/concat_1:output:0*
Index0*
T0*
_output_shapes
:4p
.linear_to_mel_weight_matrix/frame/frame_lengthConst*
_output_shapes
: *
dtype0*
value	B :n
,linear_to_mel_weight_matrix/frame/frame_stepConst*
_output_shapes
: *
dtype0*
value	B :q
&linear_to_mel_weight_matrix/frame/axisConst*
_output_shapes
: *
dtype0*
valueB :
���������q
'linear_to_mel_weight_matrix/frame/ShapeConst*
_output_shapes
:*
dtype0*
valueB:4o
,linear_to_mel_weight_matrix/frame/Size/ConstConst*
_output_shapes
: *
dtype0*
valueB h
&linear_to_mel_weight_matrix/frame/SizeConst*
_output_shapes
: *
dtype0*
value	B : q
.linear_to_mel_weight_matrix/frame/Size_1/ConstConst*
_output_shapes
: *
dtype0*
valueB j
(linear_to_mel_weight_matrix/frame/Size_1Const*
_output_shapes
: *
dtype0*
value	B : i
'linear_to_mel_weight_matrix/frame/ConstConst*
_output_shapes
: *
dtype0*
value	B : i
'linear_to_mel_weight_matrix/frame/sub/xConst*
_output_shapes
: *
dtype0*
value	B :4�
%linear_to_mel_weight_matrix/frame/subSub0linear_to_mel_weight_matrix/frame/sub/x:output:07linear_to_mel_weight_matrix/frame/frame_length:output:0*
T0*
_output_shapes
: �
*linear_to_mel_weight_matrix/frame/floordivFloorDiv)linear_to_mel_weight_matrix/frame/sub:z:05linear_to_mel_weight_matrix/frame/frame_step:output:0*
T0*
_output_shapes
: i
'linear_to_mel_weight_matrix/frame/add/xConst*
_output_shapes
: *
dtype0*
value	B :�
%linear_to_mel_weight_matrix/frame/addAddV20linear_to_mel_weight_matrix/frame/add/x:output:0.linear_to_mel_weight_matrix/frame/floordiv:z:0*
T0*
_output_shapes
: �
)linear_to_mel_weight_matrix/frame/MaximumMaximum0linear_to_mel_weight_matrix/frame/Const:output:0)linear_to_mel_weight_matrix/frame/add:z:0*
T0*
_output_shapes
: m
+linear_to_mel_weight_matrix/frame/gcd/ConstConst*
_output_shapes
: *
dtype0*
value	B :p
.linear_to_mel_weight_matrix/frame/floordiv_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
,linear_to_mel_weight_matrix/frame/floordiv_1FloorDiv7linear_to_mel_weight_matrix/frame/frame_length:output:07linear_to_mel_weight_matrix/frame/floordiv_1/y:output:0*
T0*
_output_shapes
: p
.linear_to_mel_weight_matrix/frame/floordiv_2/yConst*
_output_shapes
: *
dtype0*
value	B :�
,linear_to_mel_weight_matrix/frame/floordiv_2FloorDiv5linear_to_mel_weight_matrix/frame/frame_step:output:07linear_to_mel_weight_matrix/frame/floordiv_2/y:output:0*
T0*
_output_shapes
: t
1linear_to_mel_weight_matrix/frame/concat/values_0Const*
_output_shapes
: *
dtype0*
valueB {
1linear_to_mel_weight_matrix/frame/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:4t
1linear_to_mel_weight_matrix/frame/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB o
-linear_to_mel_weight_matrix/frame/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
(linear_to_mel_weight_matrix/frame/concatConcatV2:linear_to_mel_weight_matrix/frame/concat/values_0:output:0:linear_to_mel_weight_matrix/frame/concat/values_1:output:0:linear_to_mel_weight_matrix/frame/concat/values_2:output:06linear_to_mel_weight_matrix/frame/concat/axis:output:0*
N*
T0*
_output_shapes
:v
3linear_to_mel_weight_matrix/frame/concat_1/values_0Const*
_output_shapes
: *
dtype0*
valueB �
3linear_to_mel_weight_matrix/frame/concat_1/values_1Const*
_output_shapes
:*
dtype0*
valueB"4      v
3linear_to_mel_weight_matrix/frame/concat_1/values_2Const*
_output_shapes
: *
dtype0*
valueB q
/linear_to_mel_weight_matrix/frame/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
*linear_to_mel_weight_matrix/frame/concat_1ConcatV2<linear_to_mel_weight_matrix/frame/concat_1/values_0:output:0<linear_to_mel_weight_matrix/frame/concat_1/values_1:output:0<linear_to_mel_weight_matrix/frame/concat_1/values_2:output:08linear_to_mel_weight_matrix/frame/concat_1/axis:output:0*
N*
T0*
_output_shapes
:}
3linear_to_mel_weight_matrix/frame/zeros_like/tensorConst*
_output_shapes
:*
dtype0*
valueB:4v
,linear_to_mel_weight_matrix/frame/zeros_likeConst*
_output_shapes
:*
dtype0*
valueB: �
Alinear_to_mel_weight_matrix/frame/ones_like/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:s
1linear_to_mel_weight_matrix/frame/ones_like/ConstConst*
_output_shapes
: *
dtype0*
value	B :�
+linear_to_mel_weight_matrix/frame/ones_likeFillJlinear_to_mel_weight_matrix/frame/ones_like/Shape/shape_as_tensor:output:0:linear_to_mel_weight_matrix/frame/ones_like/Const:output:0*
T0*
_output_shapes
:�
.linear_to_mel_weight_matrix/frame/StridedSliceStridedSlice5linear_to_mel_weight_matrix/linspace_1/Slice:output:05linear_to_mel_weight_matrix/frame/zeros_like:output:01linear_to_mel_weight_matrix/frame/concat:output:04linear_to_mel_weight_matrix/frame/ones_like:output:0*
Index0*
T0*
_output_shapes
:4�
)linear_to_mel_weight_matrix/frame/ReshapeReshape7linear_to_mel_weight_matrix/frame/StridedSlice:output:03linear_to_mel_weight_matrix/frame/concat_1:output:0*
T0*
_output_shapes

:4o
-linear_to_mel_weight_matrix/frame/range/startConst*
_output_shapes
: *
dtype0*
value	B : o
-linear_to_mel_weight_matrix/frame/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
'linear_to_mel_weight_matrix/frame/rangeRange6linear_to_mel_weight_matrix/frame/range/start:output:0-linear_to_mel_weight_matrix/frame/Maximum:z:06linear_to_mel_weight_matrix/frame/range/delta:output:0*
_output_shapes
:2�
%linear_to_mel_weight_matrix/frame/mulMul0linear_to_mel_weight_matrix/frame/range:output:00linear_to_mel_weight_matrix/frame/floordiv_2:z:0*
T0*
_output_shapes
:2u
3linear_to_mel_weight_matrix/frame/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :�
1linear_to_mel_weight_matrix/frame/Reshape_1/shapePack-linear_to_mel_weight_matrix/frame/Maximum:z:0<linear_to_mel_weight_matrix/frame/Reshape_1/shape/1:output:0*
N*
T0*
_output_shapes
:�
+linear_to_mel_weight_matrix/frame/Reshape_1Reshape)linear_to_mel_weight_matrix/frame/mul:z:0:linear_to_mel_weight_matrix/frame/Reshape_1/shape:output:0*
T0*
_output_shapes

:2q
/linear_to_mel_weight_matrix/frame/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : q
/linear_to_mel_weight_matrix/frame/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
)linear_to_mel_weight_matrix/frame/range_1Range8linear_to_mel_weight_matrix/frame/range_1/start:output:00linear_to_mel_weight_matrix/frame/floordiv_1:z:08linear_to_mel_weight_matrix/frame/range_1/delta:output:0*
_output_shapes
:u
3linear_to_mel_weight_matrix/frame/Reshape_2/shape/0Const*
_output_shapes
: *
dtype0*
value	B :�
1linear_to_mel_weight_matrix/frame/Reshape_2/shapePack<linear_to_mel_weight_matrix/frame/Reshape_2/shape/0:output:00linear_to_mel_weight_matrix/frame/floordiv_1:z:0*
N*
T0*
_output_shapes
:�
+linear_to_mel_weight_matrix/frame/Reshape_2Reshape2linear_to_mel_weight_matrix/frame/range_1:output:0:linear_to_mel_weight_matrix/frame/Reshape_2/shape:output:0*
T0*
_output_shapes

:�
'linear_to_mel_weight_matrix/frame/add_1AddV24linear_to_mel_weight_matrix/frame/Reshape_1:output:04linear_to_mel_weight_matrix/frame/Reshape_2:output:0*
T0*
_output_shapes

:2l
)linear_to_mel_weight_matrix/frame/Const_1Const*
_output_shapes
: *
dtype0*
valueB l
)linear_to_mel_weight_matrix/frame/Const_2Const*
_output_shapes
: *
dtype0*
valueB �
(linear_to_mel_weight_matrix/frame/packedPack-linear_to_mel_weight_matrix/frame/Maximum:z:07linear_to_mel_weight_matrix/frame/frame_length:output:0*
N*
T0*
_output_shapes
:q
/linear_to_mel_weight_matrix/frame/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
*linear_to_mel_weight_matrix/frame/GatherV2GatherV22linear_to_mel_weight_matrix/frame/Reshape:output:0+linear_to_mel_weight_matrix/frame/add_1:z:08linear_to_mel_weight_matrix/frame/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*"
_output_shapes
:2q
/linear_to_mel_weight_matrix/frame/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
*linear_to_mel_weight_matrix/frame/concat_2ConcatV22linear_to_mel_weight_matrix/frame/Const_1:output:01linear_to_mel_weight_matrix/frame/packed:output:02linear_to_mel_weight_matrix/frame/Const_2:output:08linear_to_mel_weight_matrix/frame/concat_2/axis:output:0*
N*
T0*
_output_shapes
:�
+linear_to_mel_weight_matrix/frame/Reshape_3Reshape3linear_to_mel_weight_matrix/frame/GatherV2:output:03linear_to_mel_weight_matrix/frame/concat_2:output:0*
T0*
_output_shapes

:2m
+linear_to_mel_weight_matrix/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
!linear_to_mel_weight_matrix/splitSplit4linear_to_mel_weight_matrix/split/split_dim:output:04linear_to_mel_weight_matrix/frame/Reshape_3:output:0*
T0*2
_output_shapes 
:2:2:2*
	num_splitz
)linear_to_mel_weight_matrix/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   2   �
#linear_to_mel_weight_matrix/ReshapeReshape*linear_to_mel_weight_matrix/split:output:02linear_to_mel_weight_matrix/Reshape/shape:output:0*
T0*
_output_shapes

:2|
+linear_to_mel_weight_matrix/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   2   �
%linear_to_mel_weight_matrix/Reshape_1Reshape*linear_to_mel_weight_matrix/split:output:14linear_to_mel_weight_matrix/Reshape_1/shape:output:0*
T0*
_output_shapes

:2|
+linear_to_mel_weight_matrix/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"   2   �
%linear_to_mel_weight_matrix/Reshape_2Reshape*linear_to_mel_weight_matrix/split:output:24linear_to_mel_weight_matrix/Reshape_2/shape:output:0*
T0*
_output_shapes

:2�
linear_to_mel_weight_matrix/subSub/linear_to_mel_weight_matrix/ExpandDims:output:0,linear_to_mel_weight_matrix/Reshape:output:0*
T0*
_output_shapes
:	�2�
!linear_to_mel_weight_matrix/sub_1Sub.linear_to_mel_weight_matrix/Reshape_1:output:0,linear_to_mel_weight_matrix/Reshape:output:0*
T0*
_output_shapes

:2�
%linear_to_mel_weight_matrix/truediv_1RealDiv#linear_to_mel_weight_matrix/sub:z:0%linear_to_mel_weight_matrix/sub_1:z:0*
T0*
_output_shapes
:	�2�
!linear_to_mel_weight_matrix/sub_2Sub.linear_to_mel_weight_matrix/Reshape_2:output:0/linear_to_mel_weight_matrix/ExpandDims:output:0*
T0*
_output_shapes
:	�2�
!linear_to_mel_weight_matrix/sub_3Sub.linear_to_mel_weight_matrix/Reshape_2:output:0.linear_to_mel_weight_matrix/Reshape_1:output:0*
T0*
_output_shapes

:2�
%linear_to_mel_weight_matrix/truediv_2RealDiv%linear_to_mel_weight_matrix/sub_2:z:0%linear_to_mel_weight_matrix/sub_3:z:0*
T0*
_output_shapes
:	�2�
#linear_to_mel_weight_matrix/MinimumMinimum)linear_to_mel_weight_matrix/truediv_1:z:0)linear_to_mel_weight_matrix/truediv_2:z:0*
T0*
_output_shapes
:	�2�
#linear_to_mel_weight_matrix/MaximumMaximum*linear_to_mel_weight_matrix/Const:output:0'linear_to_mel_weight_matrix/Minimum:z:0*
T0*
_output_shapes
:	�2�
$linear_to_mel_weight_matrix/paddingsConst*
_output_shapes

:*
dtype0*)
value B"               �
linear_to_mel_weight_matrixPad'linear_to_mel_weight_matrix/Maximum:z:0-linear_to_mel_weight_matrix/paddings:output:0*
T0*
_output_shapes
:	�2q
MatMulMatMulAbs:y:0$linear_to_mel_weight_matrix:output:0*
T0*'
_output_shapes
:���������2J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5`
addAddV2MatMul:product:0add/y:output:0*
T0*'
_output_shapes
:���������2E
LogLogadd:z:0*
T0*'
_output_shapes
:���������2Y
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������p

ExpandDims
ExpandDimsLog:y:0ExpandDims/dim:output:0*
T0*+
_output_shapes
:���������2R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
ExpandDims_1
ExpandDimsExpandDims:output:0ExpandDims_1/dim:output:0*
T0*/
_output_shapes
:���������2�
StatefulPartitionedCallStatefulPartitionedCallExpandDims_1:output:0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37*3
Tin,
*2(*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*G
_read_only_resource_inputs)
'%	
 !"#$%&'*-
config_proto

CPU

GPU 2J 8� *3
f.R,
*__inference_signature_wrapper___call___753[
ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
���������{
ArgMaxArgMax StatefulPartitionedCall:output:0ArgMax/dimension:output:0*
T0*#
_output_shapes
:����������
GatherV2/paramsConst*
_output_shapes
:*
dtype0*[
valueRBPB
backgroundBdownBgoBleftBnoBoffBonBrightBstopBupByesBunknownO
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
GatherV2GatherV2GatherV2/params:output:0ArgMax:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:���������Z
IdentityIdentityArgMax:output:0^NoOp*
T0	*#
_output_shapes
:���������^

Identity_1IdentityGatherV2:output:0^NoOp*
T0*#
_output_shapes
:���������q

Identity_2Identity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:����������}:b2:b2: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:$' 

_user_specified_name2386:$& 

_user_specified_name2384:$% 

_user_specified_name2382:$$ 

_user_specified_name2380:$# 

_user_specified_name2378:$" 

_user_specified_name2376:$! 

_user_specified_name2374:$  

_user_specified_name2372:$ 

_user_specified_name2370:$ 

_user_specified_name2368:$ 

_user_specified_name2366:$ 

_user_specified_name2364:$ 

_user_specified_name2362:$ 

_user_specified_name2360:$ 

_user_specified_name2358:$ 

_user_specified_name2356:$ 

_user_specified_name2354:$ 

_user_specified_name2352:$ 

_user_specified_name2350:$ 

_user_specified_name2348:$ 

_user_specified_name2346:$ 

_user_specified_name2344:$ 

_user_specified_name2342:$ 

_user_specified_name2340:$ 

_user_specified_name2338:$ 

_user_specified_name2336:$ 

_user_specified_name2334:$ 

_user_specified_name2332:$ 

_user_specified_name2330:$
 

_user_specified_name2328:$	 

_user_specified_name2326:$ 

_user_specified_name2324:$ 

_user_specified_name2322:$ 

_user_specified_name2320:$ 

_user_specified_name2318:$ 

_user_specified_name2316:$ 

_user_specified_name2314:LH
&
_output_shapes
:b2

_user_specified_name2312:LH
&
_output_shapes
:b2

_user_specified_name2310:O K
(
_output_shapes
:����������}

_user_specified_nameinput
�
�H
__inference__traced_save_4035
file_prefix=
'read_disablecopyonread_imageinput__mean:b2C
-read_1_disablecopyonread_imageinput__variance:b24
*read_2_disablecopyonread_imageinput__count:	 C
)read_3_disablecopyonread_conv_1__kernel_1:5
'read_4_disablecopyonread_conv_1__bias_1:;
-read_5_disablecopyonread_batchnorm_1__gamma_1::
,read_6_disablecopyonread_batchnorm_1__beta_1:A
3read_7_disablecopyonread_batchnorm_1__moving_mean_1:E
7read_8_disablecopyonread_batchnorm_1__moving_variance_1:C
)read_9_disablecopyonread_conv_2__kernel_1:6
(read_10_disablecopyonread_conv_2__bias_1:<
.read_11_disablecopyonread_batchnorm_2__gamma_1:;
-read_12_disablecopyonread_batchnorm_2__beta_1:B
4read_13_disablecopyonread_batchnorm_2__moving_mean_1:F
8read_14_disablecopyonread_batchnorm_2__moving_variance_1:D
*read_15_disablecopyonread_conv_3__kernel_1:6
(read_16_disablecopyonread_conv_3__bias_1:<
.read_17_disablecopyonread_batchnorm_3__gamma_1:;
-read_18_disablecopyonread_batchnorm_3__beta_1:B
4read_19_disablecopyonread_batchnorm_3__moving_mean_1:F
8read_20_disablecopyonread_batchnorm_3__moving_variance_1:D
*read_21_disablecopyonread_conv_4__kernel_1: 6
(read_22_disablecopyonread_conv_4__bias_1: <
.read_23_disablecopyonread_batchnorm_4__gamma_1: ;
-read_24_disablecopyonread_batchnorm_4__beta_1: B
4read_25_disablecopyonread_batchnorm_4__moving_mean_1: F
8read_26_disablecopyonread_batchnorm_4__moving_variance_1: D
*read_27_disablecopyonread_conv_5__kernel_1: P6
(read_28_disablecopyonread_conv_5__bias_1:P<
.read_29_disablecopyonread_batchnorm_5__gamma_1:P;
-read_30_disablecopyonread_batchnorm_5__beta_1:PB
4read_31_disablecopyonread_batchnorm_5__moving_mean_1:PF
8read_32_disablecopyonread_batchnorm_5__moving_variance_1:PF
2read_33_disablecopyonread_lstm__lstm_cell_kernel_1:
��P
<read_34_disablecopyonread_lstm__lstm_cell_recurrent_kernel_1:
��?
0read_35_disablecopyonread_lstm__lstm_cell_bias_1:	�>
0read_36_disablecopyonread_seed_generator_state_1:;
(read_37_disablecopyonread_fc_1__kernel_1:	�4
&read_38_disablecopyonread_fc_1__bias_1:<
.read_39_disablecopyonread_seed_generator_state::
(read_40_disablecopyonread_fc_2__kernel_1:4
&read_41_disablecopyonread_fc_2__bias_1:9
+read_42_disablecopyonread_batchnorm_1__beta:B
(read_43_disablecopyonread_conv_2__kernel:9
+read_44_disablecopyonread_batchnorm_3__beta:B
(read_45_disablecopyonread_conv_4__kernel: 4
&read_46_disablecopyonread_conv_5__bias:PB
(read_47_disablecopyonread_conv_1__kernel:4
&read_48_disablecopyonread_conv_3__bias:B
(read_49_disablecopyonread_conv_5__kernel: PB
(read_50_disablecopyonread_conv_3__kernel:D
0read_51_disablecopyonread_lstm__lstm_cell_kernel:
��9
&read_52_disablecopyonread_fc_1__kernel:	�4
&read_53_disablecopyonread_conv_1__bias::
,read_54_disablecopyonread_batchnorm_2__gamma::
,read_55_disablecopyonread_batchnorm_4__gamma: :
,read_56_disablecopyonread_batchnorm_5__gamma:PN
:read_57_disablecopyonread_lstm__lstm_cell_recurrent_kernel:
��8
&read_58_disablecopyonread_fc_2__kernel:=
.read_59_disablecopyonread_lstm__lstm_cell_bias:	�2
$read_60_disablecopyonread_fc_1__bias::
,read_61_disablecopyonread_batchnorm_3__gamma:2
$read_62_disablecopyonread_fc_2__bias:9
+read_63_disablecopyonread_batchnorm_2__beta:9
+read_64_disablecopyonread_batchnorm_4__beta: 9
+read_65_disablecopyonread_batchnorm_5__beta:P:
,read_66_disablecopyonread_batchnorm_1__gamma:4
&read_67_disablecopyonread_conv_2__bias:4
&read_68_disablecopyonread_conv_4__bias: D
6read_69_disablecopyonread_batchnorm_5__moving_variance:PD
6read_70_disablecopyonread_batchnorm_2__moving_variance:D
6read_71_disablecopyonread_batchnorm_1__moving_variance:@
2read_72_disablecopyonread_batchnorm_2__moving_mean:@
2read_73_disablecopyonread_batchnorm_4__moving_mean: @
2read_74_disablecopyonread_batchnorm_5__moving_mean:PD
6read_75_disablecopyonread_batchnorm_4__moving_variance: @
2read_76_disablecopyonread_batchnorm_1__moving_mean:@
2read_77_disablecopyonread_batchnorm_3__moving_mean:D
6read_78_disablecopyonread_batchnorm_3__moving_variance:
savev2_const_2
identity_159��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_27/DisableCopyOnRead�Read_27/ReadVariableOp�Read_28/DisableCopyOnRead�Read_28/ReadVariableOp�Read_29/DisableCopyOnRead�Read_29/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_30/DisableCopyOnRead�Read_30/ReadVariableOp�Read_31/DisableCopyOnRead�Read_31/ReadVariableOp�Read_32/DisableCopyOnRead�Read_32/ReadVariableOp�Read_33/DisableCopyOnRead�Read_33/ReadVariableOp�Read_34/DisableCopyOnRead�Read_34/ReadVariableOp�Read_35/DisableCopyOnRead�Read_35/ReadVariableOp�Read_36/DisableCopyOnRead�Read_36/ReadVariableOp�Read_37/DisableCopyOnRead�Read_37/ReadVariableOp�Read_38/DisableCopyOnRead�Read_38/ReadVariableOp�Read_39/DisableCopyOnRead�Read_39/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_40/DisableCopyOnRead�Read_40/ReadVariableOp�Read_41/DisableCopyOnRead�Read_41/ReadVariableOp�Read_42/DisableCopyOnRead�Read_42/ReadVariableOp�Read_43/DisableCopyOnRead�Read_43/ReadVariableOp�Read_44/DisableCopyOnRead�Read_44/ReadVariableOp�Read_45/DisableCopyOnRead�Read_45/ReadVariableOp�Read_46/DisableCopyOnRead�Read_46/ReadVariableOp�Read_47/DisableCopyOnRead�Read_47/ReadVariableOp�Read_48/DisableCopyOnRead�Read_48/ReadVariableOp�Read_49/DisableCopyOnRead�Read_49/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_50/DisableCopyOnRead�Read_50/ReadVariableOp�Read_51/DisableCopyOnRead�Read_51/ReadVariableOp�Read_52/DisableCopyOnRead�Read_52/ReadVariableOp�Read_53/DisableCopyOnRead�Read_53/ReadVariableOp�Read_54/DisableCopyOnRead�Read_54/ReadVariableOp�Read_55/DisableCopyOnRead�Read_55/ReadVariableOp�Read_56/DisableCopyOnRead�Read_56/ReadVariableOp�Read_57/DisableCopyOnRead�Read_57/ReadVariableOp�Read_58/DisableCopyOnRead�Read_58/ReadVariableOp�Read_59/DisableCopyOnRead�Read_59/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_60/DisableCopyOnRead�Read_60/ReadVariableOp�Read_61/DisableCopyOnRead�Read_61/ReadVariableOp�Read_62/DisableCopyOnRead�Read_62/ReadVariableOp�Read_63/DisableCopyOnRead�Read_63/ReadVariableOp�Read_64/DisableCopyOnRead�Read_64/ReadVariableOp�Read_65/DisableCopyOnRead�Read_65/ReadVariableOp�Read_66/DisableCopyOnRead�Read_66/ReadVariableOp�Read_67/DisableCopyOnRead�Read_67/ReadVariableOp�Read_68/DisableCopyOnRead�Read_68/ReadVariableOp�Read_69/DisableCopyOnRead�Read_69/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_70/DisableCopyOnRead�Read_70/ReadVariableOp�Read_71/DisableCopyOnRead�Read_71/ReadVariableOp�Read_72/DisableCopyOnRead�Read_72/ReadVariableOp�Read_73/DisableCopyOnRead�Read_73/ReadVariableOp�Read_74/DisableCopyOnRead�Read_74/ReadVariableOp�Read_75/DisableCopyOnRead�Read_75/ReadVariableOp�Read_76/DisableCopyOnRead�Read_76/ReadVariableOp�Read_77/DisableCopyOnRead�Read_77/ReadVariableOp�Read_78/DisableCopyOnRead�Read_78/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: j
Read/DisableCopyOnReadDisableCopyOnRead'read_disablecopyonread_imageinput__mean*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp'read_disablecopyonread_imageinput__mean^Read/DisableCopyOnRead*"
_output_shapes
:b2*
dtype0^
IdentityIdentityRead/ReadVariableOp:value:0*
T0*"
_output_shapes
:b2e

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*"
_output_shapes
:b2r
Read_1/DisableCopyOnReadDisableCopyOnRead-read_1_disablecopyonread_imageinput__variance*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp-read_1_disablecopyonread_imageinput__variance^Read_1/DisableCopyOnRead*"
_output_shapes
:b2*
dtype0b

Identity_2IdentityRead_1/ReadVariableOp:value:0*
T0*"
_output_shapes
:b2g

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*"
_output_shapes
:b2o
Read_2/DisableCopyOnReadDisableCopyOnRead*read_2_disablecopyonread_imageinput__count*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp*read_2_disablecopyonread_imageinput__count^Read_2/DisableCopyOnRead*
_output_shapes
: *
dtype0	V

Identity_4IdentityRead_2/ReadVariableOp:value:0*
T0	*
_output_shapes
: [

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0	*
_output_shapes
: n
Read_3/DisableCopyOnReadDisableCopyOnRead)read_3_disablecopyonread_conv_1__kernel_1*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp)read_3_disablecopyonread_conv_1__kernel_1^Read_3/DisableCopyOnRead*&
_output_shapes
:*
dtype0f

Identity_6IdentityRead_3/ReadVariableOp:value:0*
T0*&
_output_shapes
:k

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*&
_output_shapes
:l
Read_4/DisableCopyOnReadDisableCopyOnRead'read_4_disablecopyonread_conv_1__bias_1*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp'read_4_disablecopyonread_conv_1__bias_1^Read_4/DisableCopyOnRead*
_output_shapes
:*
dtype0Z

Identity_8IdentityRead_4/ReadVariableOp:value:0*
T0*
_output_shapes
:_

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes
:r
Read_5/DisableCopyOnReadDisableCopyOnRead-read_5_disablecopyonread_batchnorm_1__gamma_1*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp-read_5_disablecopyonread_batchnorm_1__gamma_1^Read_5/DisableCopyOnRead*
_output_shapes
:*
dtype0[
Identity_10IdentityRead_5/ReadVariableOp:value:0*
T0*
_output_shapes
:a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:q
Read_6/DisableCopyOnReadDisableCopyOnRead,read_6_disablecopyonread_batchnorm_1__beta_1*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp,read_6_disablecopyonread_batchnorm_1__beta_1^Read_6/DisableCopyOnRead*
_output_shapes
:*
dtype0[
Identity_12IdentityRead_6/ReadVariableOp:value:0*
T0*
_output_shapes
:a
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes
:x
Read_7/DisableCopyOnReadDisableCopyOnRead3read_7_disablecopyonread_batchnorm_1__moving_mean_1*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp3read_7_disablecopyonread_batchnorm_1__moving_mean_1^Read_7/DisableCopyOnRead*
_output_shapes
:*
dtype0[
Identity_14IdentityRead_7/ReadVariableOp:value:0*
T0*
_output_shapes
:a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:|
Read_8/DisableCopyOnReadDisableCopyOnRead7read_8_disablecopyonread_batchnorm_1__moving_variance_1*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp7read_8_disablecopyonread_batchnorm_1__moving_variance_1^Read_8/DisableCopyOnRead*
_output_shapes
:*
dtype0[
Identity_16IdentityRead_8/ReadVariableOp:value:0*
T0*
_output_shapes
:a
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes
:n
Read_9/DisableCopyOnReadDisableCopyOnRead)read_9_disablecopyonread_conv_2__kernel_1*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp)read_9_disablecopyonread_conv_2__kernel_1^Read_9/DisableCopyOnRead*&
_output_shapes
:*
dtype0g
Identity_18IdentityRead_9/ReadVariableOp:value:0*
T0*&
_output_shapes
:m
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*&
_output_shapes
:n
Read_10/DisableCopyOnReadDisableCopyOnRead(read_10_disablecopyonread_conv_2__bias_1*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp(read_10_disablecopyonread_conv_2__bias_1^Read_10/DisableCopyOnRead*
_output_shapes
:*
dtype0\
Identity_20IdentityRead_10/ReadVariableOp:value:0*
T0*
_output_shapes
:a
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
:t
Read_11/DisableCopyOnReadDisableCopyOnRead.read_11_disablecopyonread_batchnorm_2__gamma_1*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp.read_11_disablecopyonread_batchnorm_2__gamma_1^Read_11/DisableCopyOnRead*
_output_shapes
:*
dtype0\
Identity_22IdentityRead_11/ReadVariableOp:value:0*
T0*
_output_shapes
:a
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
:s
Read_12/DisableCopyOnReadDisableCopyOnRead-read_12_disablecopyonread_batchnorm_2__beta_1*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp-read_12_disablecopyonread_batchnorm_2__beta_1^Read_12/DisableCopyOnRead*
_output_shapes
:*
dtype0\
Identity_24IdentityRead_12/ReadVariableOp:value:0*
T0*
_output_shapes
:a
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes
:z
Read_13/DisableCopyOnReadDisableCopyOnRead4read_13_disablecopyonread_batchnorm_2__moving_mean_1*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp4read_13_disablecopyonread_batchnorm_2__moving_mean_1^Read_13/DisableCopyOnRead*
_output_shapes
:*
dtype0\
Identity_26IdentityRead_13/ReadVariableOp:value:0*
T0*
_output_shapes
:a
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:~
Read_14/DisableCopyOnReadDisableCopyOnRead8read_14_disablecopyonread_batchnorm_2__moving_variance_1*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp8read_14_disablecopyonread_batchnorm_2__moving_variance_1^Read_14/DisableCopyOnRead*
_output_shapes
:*
dtype0\
Identity_28IdentityRead_14/ReadVariableOp:value:0*
T0*
_output_shapes
:a
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes
:p
Read_15/DisableCopyOnReadDisableCopyOnRead*read_15_disablecopyonread_conv_3__kernel_1*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp*read_15_disablecopyonread_conv_3__kernel_1^Read_15/DisableCopyOnRead*&
_output_shapes
:*
dtype0h
Identity_30IdentityRead_15/ReadVariableOp:value:0*
T0*&
_output_shapes
:m
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*&
_output_shapes
:n
Read_16/DisableCopyOnReadDisableCopyOnRead(read_16_disablecopyonread_conv_3__bias_1*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp(read_16_disablecopyonread_conv_3__bias_1^Read_16/DisableCopyOnRead*
_output_shapes
:*
dtype0\
Identity_32IdentityRead_16/ReadVariableOp:value:0*
T0*
_output_shapes
:a
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes
:t
Read_17/DisableCopyOnReadDisableCopyOnRead.read_17_disablecopyonread_batchnorm_3__gamma_1*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp.read_17_disablecopyonread_batchnorm_3__gamma_1^Read_17/DisableCopyOnRead*
_output_shapes
:*
dtype0\
Identity_34IdentityRead_17/ReadVariableOp:value:0*
T0*
_output_shapes
:a
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
:s
Read_18/DisableCopyOnReadDisableCopyOnRead-read_18_disablecopyonread_batchnorm_3__beta_1*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp-read_18_disablecopyonread_batchnorm_3__beta_1^Read_18/DisableCopyOnRead*
_output_shapes
:*
dtype0\
Identity_36IdentityRead_18/ReadVariableOp:value:0*
T0*
_output_shapes
:a
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes
:z
Read_19/DisableCopyOnReadDisableCopyOnRead4read_19_disablecopyonread_batchnorm_3__moving_mean_1*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp4read_19_disablecopyonread_batchnorm_3__moving_mean_1^Read_19/DisableCopyOnRead*
_output_shapes
:*
dtype0\
Identity_38IdentityRead_19/ReadVariableOp:value:0*
T0*
_output_shapes
:a
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
:~
Read_20/DisableCopyOnReadDisableCopyOnRead8read_20_disablecopyonread_batchnorm_3__moving_variance_1*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp8read_20_disablecopyonread_batchnorm_3__moving_variance_1^Read_20/DisableCopyOnRead*
_output_shapes
:*
dtype0\
Identity_40IdentityRead_20/ReadVariableOp:value:0*
T0*
_output_shapes
:a
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes
:p
Read_21/DisableCopyOnReadDisableCopyOnRead*read_21_disablecopyonread_conv_4__kernel_1*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp*read_21_disablecopyonread_conv_4__kernel_1^Read_21/DisableCopyOnRead*&
_output_shapes
: *
dtype0h
Identity_42IdentityRead_21/ReadVariableOp:value:0*
T0*&
_output_shapes
: m
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*&
_output_shapes
: n
Read_22/DisableCopyOnReadDisableCopyOnRead(read_22_disablecopyonread_conv_4__bias_1*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp(read_22_disablecopyonread_conv_4__bias_1^Read_22/DisableCopyOnRead*
_output_shapes
: *
dtype0\
Identity_44IdentityRead_22/ReadVariableOp:value:0*
T0*
_output_shapes
: a
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_23/DisableCopyOnReadDisableCopyOnRead.read_23_disablecopyonread_batchnorm_4__gamma_1*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp.read_23_disablecopyonread_batchnorm_4__gamma_1^Read_23/DisableCopyOnRead*
_output_shapes
: *
dtype0\
Identity_46IdentityRead_23/ReadVariableOp:value:0*
T0*
_output_shapes
: a
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes
: s
Read_24/DisableCopyOnReadDisableCopyOnRead-read_24_disablecopyonread_batchnorm_4__beta_1*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOp-read_24_disablecopyonread_batchnorm_4__beta_1^Read_24/DisableCopyOnRead*
_output_shapes
: *
dtype0\
Identity_48IdentityRead_24/ReadVariableOp:value:0*
T0*
_output_shapes
: a
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes
: z
Read_25/DisableCopyOnReadDisableCopyOnRead4read_25_disablecopyonread_batchnorm_4__moving_mean_1*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOp4read_25_disablecopyonread_batchnorm_4__moving_mean_1^Read_25/DisableCopyOnRead*
_output_shapes
: *
dtype0\
Identity_50IdentityRead_25/ReadVariableOp:value:0*
T0*
_output_shapes
: a
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes
: ~
Read_26/DisableCopyOnReadDisableCopyOnRead8read_26_disablecopyonread_batchnorm_4__moving_variance_1*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOp8read_26_disablecopyonread_batchnorm_4__moving_variance_1^Read_26/DisableCopyOnRead*
_output_shapes
: *
dtype0\
Identity_52IdentityRead_26/ReadVariableOp:value:0*
T0*
_output_shapes
: a
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes
: p
Read_27/DisableCopyOnReadDisableCopyOnRead*read_27_disablecopyonread_conv_5__kernel_1*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOp*read_27_disablecopyonread_conv_5__kernel_1^Read_27/DisableCopyOnRead*&
_output_shapes
: P*
dtype0h
Identity_54IdentityRead_27/ReadVariableOp:value:0*
T0*&
_output_shapes
: Pm
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*&
_output_shapes
: Pn
Read_28/DisableCopyOnReadDisableCopyOnRead(read_28_disablecopyonread_conv_5__bias_1*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOp(read_28_disablecopyonread_conv_5__bias_1^Read_28/DisableCopyOnRead*
_output_shapes
:P*
dtype0\
Identity_56IdentityRead_28/ReadVariableOp:value:0*
T0*
_output_shapes
:Pa
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*
_output_shapes
:Pt
Read_29/DisableCopyOnReadDisableCopyOnRead.read_29_disablecopyonread_batchnorm_5__gamma_1*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOp.read_29_disablecopyonread_batchnorm_5__gamma_1^Read_29/DisableCopyOnRead*
_output_shapes
:P*
dtype0\
Identity_58IdentityRead_29/ReadVariableOp:value:0*
T0*
_output_shapes
:Pa
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes
:Ps
Read_30/DisableCopyOnReadDisableCopyOnRead-read_30_disablecopyonread_batchnorm_5__beta_1*
_output_shapes
 �
Read_30/ReadVariableOpReadVariableOp-read_30_disablecopyonread_batchnorm_5__beta_1^Read_30/DisableCopyOnRead*
_output_shapes
:P*
dtype0\
Identity_60IdentityRead_30/ReadVariableOp:value:0*
T0*
_output_shapes
:Pa
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*
_output_shapes
:Pz
Read_31/DisableCopyOnReadDisableCopyOnRead4read_31_disablecopyonread_batchnorm_5__moving_mean_1*
_output_shapes
 �
Read_31/ReadVariableOpReadVariableOp4read_31_disablecopyonread_batchnorm_5__moving_mean_1^Read_31/DisableCopyOnRead*
_output_shapes
:P*
dtype0\
Identity_62IdentityRead_31/ReadVariableOp:value:0*
T0*
_output_shapes
:Pa
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*
_output_shapes
:P~
Read_32/DisableCopyOnReadDisableCopyOnRead8read_32_disablecopyonread_batchnorm_5__moving_variance_1*
_output_shapes
 �
Read_32/ReadVariableOpReadVariableOp8read_32_disablecopyonread_batchnorm_5__moving_variance_1^Read_32/DisableCopyOnRead*
_output_shapes
:P*
dtype0\
Identity_64IdentityRead_32/ReadVariableOp:value:0*
T0*
_output_shapes
:Pa
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*
_output_shapes
:Px
Read_33/DisableCopyOnReadDisableCopyOnRead2read_33_disablecopyonread_lstm__lstm_cell_kernel_1*
_output_shapes
 �
Read_33/ReadVariableOpReadVariableOp2read_33_disablecopyonread_lstm__lstm_cell_kernel_1^Read_33/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0b
Identity_66IdentityRead_33/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��g
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_34/DisableCopyOnReadDisableCopyOnRead<read_34_disablecopyonread_lstm__lstm_cell_recurrent_kernel_1*
_output_shapes
 �
Read_34/ReadVariableOpReadVariableOp<read_34_disablecopyonread_lstm__lstm_cell_recurrent_kernel_1^Read_34/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0b
Identity_68IdentityRead_34/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��g
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��v
Read_35/DisableCopyOnReadDisableCopyOnRead0read_35_disablecopyonread_lstm__lstm_cell_bias_1*
_output_shapes
 �
Read_35/ReadVariableOpReadVariableOp0read_35_disablecopyonread_lstm__lstm_cell_bias_1^Read_35/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_70IdentityRead_35/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*
_output_shapes	
:�v
Read_36/DisableCopyOnReadDisableCopyOnRead0read_36_disablecopyonread_seed_generator_state_1*
_output_shapes
 �
Read_36/ReadVariableOpReadVariableOp0read_36_disablecopyonread_seed_generator_state_1^Read_36/DisableCopyOnRead*
_output_shapes
:*
dtype0\
Identity_72IdentityRead_36/ReadVariableOp:value:0*
T0*
_output_shapes
:a
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0*
_output_shapes
:n
Read_37/DisableCopyOnReadDisableCopyOnRead(read_37_disablecopyonread_fc_1__kernel_1*
_output_shapes
 �
Read_37/ReadVariableOpReadVariableOp(read_37_disablecopyonread_fc_1__kernel_1^Read_37/DisableCopyOnRead*
_output_shapes
:	�*
dtype0a
Identity_74IdentityRead_37/ReadVariableOp:value:0*
T0*
_output_shapes
:	�f
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0*
_output_shapes
:	�l
Read_38/DisableCopyOnReadDisableCopyOnRead&read_38_disablecopyonread_fc_1__bias_1*
_output_shapes
 �
Read_38/ReadVariableOpReadVariableOp&read_38_disablecopyonread_fc_1__bias_1^Read_38/DisableCopyOnRead*
_output_shapes
:*
dtype0\
Identity_76IdentityRead_38/ReadVariableOp:value:0*
T0*
_output_shapes
:a
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0*
_output_shapes
:t
Read_39/DisableCopyOnReadDisableCopyOnRead.read_39_disablecopyonread_seed_generator_state*
_output_shapes
 �
Read_39/ReadVariableOpReadVariableOp.read_39_disablecopyonread_seed_generator_state^Read_39/DisableCopyOnRead*
_output_shapes
:*
dtype0\
Identity_78IdentityRead_39/ReadVariableOp:value:0*
T0*
_output_shapes
:a
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0*
_output_shapes
:n
Read_40/DisableCopyOnReadDisableCopyOnRead(read_40_disablecopyonread_fc_2__kernel_1*
_output_shapes
 �
Read_40/ReadVariableOpReadVariableOp(read_40_disablecopyonread_fc_2__kernel_1^Read_40/DisableCopyOnRead*
_output_shapes

:*
dtype0`
Identity_80IdentityRead_40/ReadVariableOp:value:0*
T0*
_output_shapes

:e
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0*
_output_shapes

:l
Read_41/DisableCopyOnReadDisableCopyOnRead&read_41_disablecopyonread_fc_2__bias_1*
_output_shapes
 �
Read_41/ReadVariableOpReadVariableOp&read_41_disablecopyonread_fc_2__bias_1^Read_41/DisableCopyOnRead*
_output_shapes
:*
dtype0\
Identity_82IdentityRead_41/ReadVariableOp:value:0*
T0*
_output_shapes
:a
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0*
_output_shapes
:q
Read_42/DisableCopyOnReadDisableCopyOnRead+read_42_disablecopyonread_batchnorm_1__beta*
_output_shapes
 �
Read_42/ReadVariableOpReadVariableOp+read_42_disablecopyonread_batchnorm_1__beta^Read_42/DisableCopyOnRead*
_output_shapes
:*
dtype0\
Identity_84IdentityRead_42/ReadVariableOp:value:0*
T0*
_output_shapes
:a
Identity_85IdentityIdentity_84:output:0"/device:CPU:0*
T0*
_output_shapes
:n
Read_43/DisableCopyOnReadDisableCopyOnRead(read_43_disablecopyonread_conv_2__kernel*
_output_shapes
 �
Read_43/ReadVariableOpReadVariableOp(read_43_disablecopyonread_conv_2__kernel^Read_43/DisableCopyOnRead*&
_output_shapes
:*
dtype0h
Identity_86IdentityRead_43/ReadVariableOp:value:0*
T0*&
_output_shapes
:m
Identity_87IdentityIdentity_86:output:0"/device:CPU:0*
T0*&
_output_shapes
:q
Read_44/DisableCopyOnReadDisableCopyOnRead+read_44_disablecopyonread_batchnorm_3__beta*
_output_shapes
 �
Read_44/ReadVariableOpReadVariableOp+read_44_disablecopyonread_batchnorm_3__beta^Read_44/DisableCopyOnRead*
_output_shapes
:*
dtype0\
Identity_88IdentityRead_44/ReadVariableOp:value:0*
T0*
_output_shapes
:a
Identity_89IdentityIdentity_88:output:0"/device:CPU:0*
T0*
_output_shapes
:n
Read_45/DisableCopyOnReadDisableCopyOnRead(read_45_disablecopyonread_conv_4__kernel*
_output_shapes
 �
Read_45/ReadVariableOpReadVariableOp(read_45_disablecopyonread_conv_4__kernel^Read_45/DisableCopyOnRead*&
_output_shapes
: *
dtype0h
Identity_90IdentityRead_45/ReadVariableOp:value:0*
T0*&
_output_shapes
: m
Identity_91IdentityIdentity_90:output:0"/device:CPU:0*
T0*&
_output_shapes
: l
Read_46/DisableCopyOnReadDisableCopyOnRead&read_46_disablecopyonread_conv_5__bias*
_output_shapes
 �
Read_46/ReadVariableOpReadVariableOp&read_46_disablecopyonread_conv_5__bias^Read_46/DisableCopyOnRead*
_output_shapes
:P*
dtype0\
Identity_92IdentityRead_46/ReadVariableOp:value:0*
T0*
_output_shapes
:Pa
Identity_93IdentityIdentity_92:output:0"/device:CPU:0*
T0*
_output_shapes
:Pn
Read_47/DisableCopyOnReadDisableCopyOnRead(read_47_disablecopyonread_conv_1__kernel*
_output_shapes
 �
Read_47/ReadVariableOpReadVariableOp(read_47_disablecopyonread_conv_1__kernel^Read_47/DisableCopyOnRead*&
_output_shapes
:*
dtype0h
Identity_94IdentityRead_47/ReadVariableOp:value:0*
T0*&
_output_shapes
:m
Identity_95IdentityIdentity_94:output:0"/device:CPU:0*
T0*&
_output_shapes
:l
Read_48/DisableCopyOnReadDisableCopyOnRead&read_48_disablecopyonread_conv_3__bias*
_output_shapes
 �
Read_48/ReadVariableOpReadVariableOp&read_48_disablecopyonread_conv_3__bias^Read_48/DisableCopyOnRead*
_output_shapes
:*
dtype0\
Identity_96IdentityRead_48/ReadVariableOp:value:0*
T0*
_output_shapes
:a
Identity_97IdentityIdentity_96:output:0"/device:CPU:0*
T0*
_output_shapes
:n
Read_49/DisableCopyOnReadDisableCopyOnRead(read_49_disablecopyonread_conv_5__kernel*
_output_shapes
 �
Read_49/ReadVariableOpReadVariableOp(read_49_disablecopyonread_conv_5__kernel^Read_49/DisableCopyOnRead*&
_output_shapes
: P*
dtype0h
Identity_98IdentityRead_49/ReadVariableOp:value:0*
T0*&
_output_shapes
: Pm
Identity_99IdentityIdentity_98:output:0"/device:CPU:0*
T0*&
_output_shapes
: Pn
Read_50/DisableCopyOnReadDisableCopyOnRead(read_50_disablecopyonread_conv_3__kernel*
_output_shapes
 �
Read_50/ReadVariableOpReadVariableOp(read_50_disablecopyonread_conv_3__kernel^Read_50/DisableCopyOnRead*&
_output_shapes
:*
dtype0i
Identity_100IdentityRead_50/ReadVariableOp:value:0*
T0*&
_output_shapes
:o
Identity_101IdentityIdentity_100:output:0"/device:CPU:0*
T0*&
_output_shapes
:v
Read_51/DisableCopyOnReadDisableCopyOnRead0read_51_disablecopyonread_lstm__lstm_cell_kernel*
_output_shapes
 �
Read_51/ReadVariableOpReadVariableOp0read_51_disablecopyonread_lstm__lstm_cell_kernel^Read_51/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0c
Identity_102IdentityRead_51/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��i
Identity_103IdentityIdentity_102:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��l
Read_52/DisableCopyOnReadDisableCopyOnRead&read_52_disablecopyonread_fc_1__kernel*
_output_shapes
 �
Read_52/ReadVariableOpReadVariableOp&read_52_disablecopyonread_fc_1__kernel^Read_52/DisableCopyOnRead*
_output_shapes
:	�*
dtype0b
Identity_104IdentityRead_52/ReadVariableOp:value:0*
T0*
_output_shapes
:	�h
Identity_105IdentityIdentity_104:output:0"/device:CPU:0*
T0*
_output_shapes
:	�l
Read_53/DisableCopyOnReadDisableCopyOnRead&read_53_disablecopyonread_conv_1__bias*
_output_shapes
 �
Read_53/ReadVariableOpReadVariableOp&read_53_disablecopyonread_conv_1__bias^Read_53/DisableCopyOnRead*
_output_shapes
:*
dtype0]
Identity_106IdentityRead_53/ReadVariableOp:value:0*
T0*
_output_shapes
:c
Identity_107IdentityIdentity_106:output:0"/device:CPU:0*
T0*
_output_shapes
:r
Read_54/DisableCopyOnReadDisableCopyOnRead,read_54_disablecopyonread_batchnorm_2__gamma*
_output_shapes
 �
Read_54/ReadVariableOpReadVariableOp,read_54_disablecopyonread_batchnorm_2__gamma^Read_54/DisableCopyOnRead*
_output_shapes
:*
dtype0]
Identity_108IdentityRead_54/ReadVariableOp:value:0*
T0*
_output_shapes
:c
Identity_109IdentityIdentity_108:output:0"/device:CPU:0*
T0*
_output_shapes
:r
Read_55/DisableCopyOnReadDisableCopyOnRead,read_55_disablecopyonread_batchnorm_4__gamma*
_output_shapes
 �
Read_55/ReadVariableOpReadVariableOp,read_55_disablecopyonread_batchnorm_4__gamma^Read_55/DisableCopyOnRead*
_output_shapes
: *
dtype0]
Identity_110IdentityRead_55/ReadVariableOp:value:0*
T0*
_output_shapes
: c
Identity_111IdentityIdentity_110:output:0"/device:CPU:0*
T0*
_output_shapes
: r
Read_56/DisableCopyOnReadDisableCopyOnRead,read_56_disablecopyonread_batchnorm_5__gamma*
_output_shapes
 �
Read_56/ReadVariableOpReadVariableOp,read_56_disablecopyonread_batchnorm_5__gamma^Read_56/DisableCopyOnRead*
_output_shapes
:P*
dtype0]
Identity_112IdentityRead_56/ReadVariableOp:value:0*
T0*
_output_shapes
:Pc
Identity_113IdentityIdentity_112:output:0"/device:CPU:0*
T0*
_output_shapes
:P�
Read_57/DisableCopyOnReadDisableCopyOnRead:read_57_disablecopyonread_lstm__lstm_cell_recurrent_kernel*
_output_shapes
 �
Read_57/ReadVariableOpReadVariableOp:read_57_disablecopyonread_lstm__lstm_cell_recurrent_kernel^Read_57/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0c
Identity_114IdentityRead_57/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��i
Identity_115IdentityIdentity_114:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��l
Read_58/DisableCopyOnReadDisableCopyOnRead&read_58_disablecopyonread_fc_2__kernel*
_output_shapes
 �
Read_58/ReadVariableOpReadVariableOp&read_58_disablecopyonread_fc_2__kernel^Read_58/DisableCopyOnRead*
_output_shapes

:*
dtype0a
Identity_116IdentityRead_58/ReadVariableOp:value:0*
T0*
_output_shapes

:g
Identity_117IdentityIdentity_116:output:0"/device:CPU:0*
T0*
_output_shapes

:t
Read_59/DisableCopyOnReadDisableCopyOnRead.read_59_disablecopyonread_lstm__lstm_cell_bias*
_output_shapes
 �
Read_59/ReadVariableOpReadVariableOp.read_59_disablecopyonread_lstm__lstm_cell_bias^Read_59/DisableCopyOnRead*
_output_shapes	
:�*
dtype0^
Identity_118IdentityRead_59/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_119IdentityIdentity_118:output:0"/device:CPU:0*
T0*
_output_shapes	
:�j
Read_60/DisableCopyOnReadDisableCopyOnRead$read_60_disablecopyonread_fc_1__bias*
_output_shapes
 �
Read_60/ReadVariableOpReadVariableOp$read_60_disablecopyonread_fc_1__bias^Read_60/DisableCopyOnRead*
_output_shapes
:*
dtype0]
Identity_120IdentityRead_60/ReadVariableOp:value:0*
T0*
_output_shapes
:c
Identity_121IdentityIdentity_120:output:0"/device:CPU:0*
T0*
_output_shapes
:r
Read_61/DisableCopyOnReadDisableCopyOnRead,read_61_disablecopyonread_batchnorm_3__gamma*
_output_shapes
 �
Read_61/ReadVariableOpReadVariableOp,read_61_disablecopyonread_batchnorm_3__gamma^Read_61/DisableCopyOnRead*
_output_shapes
:*
dtype0]
Identity_122IdentityRead_61/ReadVariableOp:value:0*
T0*
_output_shapes
:c
Identity_123IdentityIdentity_122:output:0"/device:CPU:0*
T0*
_output_shapes
:j
Read_62/DisableCopyOnReadDisableCopyOnRead$read_62_disablecopyonread_fc_2__bias*
_output_shapes
 �
Read_62/ReadVariableOpReadVariableOp$read_62_disablecopyonread_fc_2__bias^Read_62/DisableCopyOnRead*
_output_shapes
:*
dtype0]
Identity_124IdentityRead_62/ReadVariableOp:value:0*
T0*
_output_shapes
:c
Identity_125IdentityIdentity_124:output:0"/device:CPU:0*
T0*
_output_shapes
:q
Read_63/DisableCopyOnReadDisableCopyOnRead+read_63_disablecopyonread_batchnorm_2__beta*
_output_shapes
 �
Read_63/ReadVariableOpReadVariableOp+read_63_disablecopyonread_batchnorm_2__beta^Read_63/DisableCopyOnRead*
_output_shapes
:*
dtype0]
Identity_126IdentityRead_63/ReadVariableOp:value:0*
T0*
_output_shapes
:c
Identity_127IdentityIdentity_126:output:0"/device:CPU:0*
T0*
_output_shapes
:q
Read_64/DisableCopyOnReadDisableCopyOnRead+read_64_disablecopyonread_batchnorm_4__beta*
_output_shapes
 �
Read_64/ReadVariableOpReadVariableOp+read_64_disablecopyonread_batchnorm_4__beta^Read_64/DisableCopyOnRead*
_output_shapes
: *
dtype0]
Identity_128IdentityRead_64/ReadVariableOp:value:0*
T0*
_output_shapes
: c
Identity_129IdentityIdentity_128:output:0"/device:CPU:0*
T0*
_output_shapes
: q
Read_65/DisableCopyOnReadDisableCopyOnRead+read_65_disablecopyonread_batchnorm_5__beta*
_output_shapes
 �
Read_65/ReadVariableOpReadVariableOp+read_65_disablecopyonread_batchnorm_5__beta^Read_65/DisableCopyOnRead*
_output_shapes
:P*
dtype0]
Identity_130IdentityRead_65/ReadVariableOp:value:0*
T0*
_output_shapes
:Pc
Identity_131IdentityIdentity_130:output:0"/device:CPU:0*
T0*
_output_shapes
:Pr
Read_66/DisableCopyOnReadDisableCopyOnRead,read_66_disablecopyonread_batchnorm_1__gamma*
_output_shapes
 �
Read_66/ReadVariableOpReadVariableOp,read_66_disablecopyonread_batchnorm_1__gamma^Read_66/DisableCopyOnRead*
_output_shapes
:*
dtype0]
Identity_132IdentityRead_66/ReadVariableOp:value:0*
T0*
_output_shapes
:c
Identity_133IdentityIdentity_132:output:0"/device:CPU:0*
T0*
_output_shapes
:l
Read_67/DisableCopyOnReadDisableCopyOnRead&read_67_disablecopyonread_conv_2__bias*
_output_shapes
 �
Read_67/ReadVariableOpReadVariableOp&read_67_disablecopyonread_conv_2__bias^Read_67/DisableCopyOnRead*
_output_shapes
:*
dtype0]
Identity_134IdentityRead_67/ReadVariableOp:value:0*
T0*
_output_shapes
:c
Identity_135IdentityIdentity_134:output:0"/device:CPU:0*
T0*
_output_shapes
:l
Read_68/DisableCopyOnReadDisableCopyOnRead&read_68_disablecopyonread_conv_4__bias*
_output_shapes
 �
Read_68/ReadVariableOpReadVariableOp&read_68_disablecopyonread_conv_4__bias^Read_68/DisableCopyOnRead*
_output_shapes
: *
dtype0]
Identity_136IdentityRead_68/ReadVariableOp:value:0*
T0*
_output_shapes
: c
Identity_137IdentityIdentity_136:output:0"/device:CPU:0*
T0*
_output_shapes
: |
Read_69/DisableCopyOnReadDisableCopyOnRead6read_69_disablecopyonread_batchnorm_5__moving_variance*
_output_shapes
 �
Read_69/ReadVariableOpReadVariableOp6read_69_disablecopyonread_batchnorm_5__moving_variance^Read_69/DisableCopyOnRead*
_output_shapes
:P*
dtype0]
Identity_138IdentityRead_69/ReadVariableOp:value:0*
T0*
_output_shapes
:Pc
Identity_139IdentityIdentity_138:output:0"/device:CPU:0*
T0*
_output_shapes
:P|
Read_70/DisableCopyOnReadDisableCopyOnRead6read_70_disablecopyonread_batchnorm_2__moving_variance*
_output_shapes
 �
Read_70/ReadVariableOpReadVariableOp6read_70_disablecopyonread_batchnorm_2__moving_variance^Read_70/DisableCopyOnRead*
_output_shapes
:*
dtype0]
Identity_140IdentityRead_70/ReadVariableOp:value:0*
T0*
_output_shapes
:c
Identity_141IdentityIdentity_140:output:0"/device:CPU:0*
T0*
_output_shapes
:|
Read_71/DisableCopyOnReadDisableCopyOnRead6read_71_disablecopyonread_batchnorm_1__moving_variance*
_output_shapes
 �
Read_71/ReadVariableOpReadVariableOp6read_71_disablecopyonread_batchnorm_1__moving_variance^Read_71/DisableCopyOnRead*
_output_shapes
:*
dtype0]
Identity_142IdentityRead_71/ReadVariableOp:value:0*
T0*
_output_shapes
:c
Identity_143IdentityIdentity_142:output:0"/device:CPU:0*
T0*
_output_shapes
:x
Read_72/DisableCopyOnReadDisableCopyOnRead2read_72_disablecopyonread_batchnorm_2__moving_mean*
_output_shapes
 �
Read_72/ReadVariableOpReadVariableOp2read_72_disablecopyonread_batchnorm_2__moving_mean^Read_72/DisableCopyOnRead*
_output_shapes
:*
dtype0]
Identity_144IdentityRead_72/ReadVariableOp:value:0*
T0*
_output_shapes
:c
Identity_145IdentityIdentity_144:output:0"/device:CPU:0*
T0*
_output_shapes
:x
Read_73/DisableCopyOnReadDisableCopyOnRead2read_73_disablecopyonread_batchnorm_4__moving_mean*
_output_shapes
 �
Read_73/ReadVariableOpReadVariableOp2read_73_disablecopyonread_batchnorm_4__moving_mean^Read_73/DisableCopyOnRead*
_output_shapes
: *
dtype0]
Identity_146IdentityRead_73/ReadVariableOp:value:0*
T0*
_output_shapes
: c
Identity_147IdentityIdentity_146:output:0"/device:CPU:0*
T0*
_output_shapes
: x
Read_74/DisableCopyOnReadDisableCopyOnRead2read_74_disablecopyonread_batchnorm_5__moving_mean*
_output_shapes
 �
Read_74/ReadVariableOpReadVariableOp2read_74_disablecopyonread_batchnorm_5__moving_mean^Read_74/DisableCopyOnRead*
_output_shapes
:P*
dtype0]
Identity_148IdentityRead_74/ReadVariableOp:value:0*
T0*
_output_shapes
:Pc
Identity_149IdentityIdentity_148:output:0"/device:CPU:0*
T0*
_output_shapes
:P|
Read_75/DisableCopyOnReadDisableCopyOnRead6read_75_disablecopyonread_batchnorm_4__moving_variance*
_output_shapes
 �
Read_75/ReadVariableOpReadVariableOp6read_75_disablecopyonread_batchnorm_4__moving_variance^Read_75/DisableCopyOnRead*
_output_shapes
: *
dtype0]
Identity_150IdentityRead_75/ReadVariableOp:value:0*
T0*
_output_shapes
: c
Identity_151IdentityIdentity_150:output:0"/device:CPU:0*
T0*
_output_shapes
: x
Read_76/DisableCopyOnReadDisableCopyOnRead2read_76_disablecopyonread_batchnorm_1__moving_mean*
_output_shapes
 �
Read_76/ReadVariableOpReadVariableOp2read_76_disablecopyonread_batchnorm_1__moving_mean^Read_76/DisableCopyOnRead*
_output_shapes
:*
dtype0]
Identity_152IdentityRead_76/ReadVariableOp:value:0*
T0*
_output_shapes
:c
Identity_153IdentityIdentity_152:output:0"/device:CPU:0*
T0*
_output_shapes
:x
Read_77/DisableCopyOnReadDisableCopyOnRead2read_77_disablecopyonread_batchnorm_3__moving_mean*
_output_shapes
 �
Read_77/ReadVariableOpReadVariableOp2read_77_disablecopyonread_batchnorm_3__moving_mean^Read_77/DisableCopyOnRead*
_output_shapes
:*
dtype0]
Identity_154IdentityRead_77/ReadVariableOp:value:0*
T0*
_output_shapes
:c
Identity_155IdentityIdentity_154:output:0"/device:CPU:0*
T0*
_output_shapes
:|
Read_78/DisableCopyOnReadDisableCopyOnRead6read_78_disablecopyonread_batchnorm_3__moving_variance*
_output_shapes
 �
Read_78/ReadVariableOpReadVariableOp6read_78_disablecopyonread_batchnorm_3__moving_variance^Read_78/DisableCopyOnRead*
_output_shapes
:*
dtype0]
Identity_156IdentityRead_78/ReadVariableOp:value:0*
T0*
_output_shapes
:c
Identity_157IdentityIdentity_156:output:0"/device:CPU:0*
T0*
_output_shapes
:L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:P*
dtype0*�
value�B�PB,model/variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model/variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model/variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model/variables/3/.ATTRIBUTES/VARIABLE_VALUEB,model/variables/4/.ATTRIBUTES/VARIABLE_VALUEB,model/variables/5/.ATTRIBUTES/VARIABLE_VALUEB,model/variables/6/.ATTRIBUTES/VARIABLE_VALUEB,model/variables/7/.ATTRIBUTES/VARIABLE_VALUEB,model/variables/8/.ATTRIBUTES/VARIABLE_VALUEB,model/variables/9/.ATTRIBUTES/VARIABLE_VALUEB-model/variables/10/.ATTRIBUTES/VARIABLE_VALUEB-model/variables/11/.ATTRIBUTES/VARIABLE_VALUEB-model/variables/12/.ATTRIBUTES/VARIABLE_VALUEB-model/variables/13/.ATTRIBUTES/VARIABLE_VALUEB-model/variables/14/.ATTRIBUTES/VARIABLE_VALUEB-model/variables/15/.ATTRIBUTES/VARIABLE_VALUEB-model/variables/16/.ATTRIBUTES/VARIABLE_VALUEB-model/variables/17/.ATTRIBUTES/VARIABLE_VALUEB-model/variables/18/.ATTRIBUTES/VARIABLE_VALUEB-model/variables/19/.ATTRIBUTES/VARIABLE_VALUEB-model/variables/20/.ATTRIBUTES/VARIABLE_VALUEB-model/variables/21/.ATTRIBUTES/VARIABLE_VALUEB-model/variables/22/.ATTRIBUTES/VARIABLE_VALUEB-model/variables/23/.ATTRIBUTES/VARIABLE_VALUEB-model/variables/24/.ATTRIBUTES/VARIABLE_VALUEB-model/variables/25/.ATTRIBUTES/VARIABLE_VALUEB-model/variables/26/.ATTRIBUTES/VARIABLE_VALUEB-model/variables/27/.ATTRIBUTES/VARIABLE_VALUEB-model/variables/28/.ATTRIBUTES/VARIABLE_VALUEB-model/variables/29/.ATTRIBUTES/VARIABLE_VALUEB-model/variables/30/.ATTRIBUTES/VARIABLE_VALUEB-model/variables/31/.ATTRIBUTES/VARIABLE_VALUEB-model/variables/32/.ATTRIBUTES/VARIABLE_VALUEB-model/variables/33/.ATTRIBUTES/VARIABLE_VALUEB-model/variables/34/.ATTRIBUTES/VARIABLE_VALUEB-model/variables/35/.ATTRIBUTES/VARIABLE_VALUEB-model/variables/36/.ATTRIBUTES/VARIABLE_VALUEB-model/variables/37/.ATTRIBUTES/VARIABLE_VALUEB-model/variables/38/.ATTRIBUTES/VARIABLE_VALUEB-model/variables/39/.ATTRIBUTES/VARIABLE_VALUEB-model/variables/40/.ATTRIBUTES/VARIABLE_VALUEB-model/variables/41/.ATTRIBUTES/VARIABLE_VALUEB1model/_all_variables/0/.ATTRIBUTES/VARIABLE_VALUEB1model/_all_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1model/_all_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1model/_all_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1model/_all_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1model/_all_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1model/_all_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1model/_all_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1model/_all_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1model/_all_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2model/_all_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2model/_all_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2model/_all_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2model/_all_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2model/_all_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2model/_all_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2model/_all_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2model/_all_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2model/_all_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2model/_all_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2model/_all_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2model/_all_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2model/_all_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2model/_all_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2model/_all_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2model/_all_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2model/_all_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2model/_all_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2model/_all_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2model/_all_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2model/_all_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2model/_all_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2model/_all_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2model/_all_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2model/_all_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2model/_all_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2model/_all_variables/36/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:P*
dtype0*�
value�B�PB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0Identity_85:output:0Identity_87:output:0Identity_89:output:0Identity_91:output:0Identity_93:output:0Identity_95:output:0Identity_97:output:0Identity_99:output:0Identity_101:output:0Identity_103:output:0Identity_105:output:0Identity_107:output:0Identity_109:output:0Identity_111:output:0Identity_113:output:0Identity_115:output:0Identity_117:output:0Identity_119:output:0Identity_121:output:0Identity_123:output:0Identity_125:output:0Identity_127:output:0Identity_129:output:0Identity_131:output:0Identity_133:output:0Identity_135:output:0Identity_137:output:0Identity_139:output:0Identity_141:output:0Identity_143:output:0Identity_145:output:0Identity_147:output:0Identity_149:output:0Identity_151:output:0Identity_153:output:0Identity_155:output:0Identity_157:output:0savev2_const_2"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *^
dtypesT
R2P	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 j
Identity_158Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: W
Identity_159IdentityIdentity_158:output:0^NoOp*
T0*
_output_shapes
: � 
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_42/DisableCopyOnRead^Read_42/ReadVariableOp^Read_43/DisableCopyOnRead^Read_43/ReadVariableOp^Read_44/DisableCopyOnRead^Read_44/ReadVariableOp^Read_45/DisableCopyOnRead^Read_45/ReadVariableOp^Read_46/DisableCopyOnRead^Read_46/ReadVariableOp^Read_47/DisableCopyOnRead^Read_47/ReadVariableOp^Read_48/DisableCopyOnRead^Read_48/ReadVariableOp^Read_49/DisableCopyOnRead^Read_49/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_50/DisableCopyOnRead^Read_50/ReadVariableOp^Read_51/DisableCopyOnRead^Read_51/ReadVariableOp^Read_52/DisableCopyOnRead^Read_52/ReadVariableOp^Read_53/DisableCopyOnRead^Read_53/ReadVariableOp^Read_54/DisableCopyOnRead^Read_54/ReadVariableOp^Read_55/DisableCopyOnRead^Read_55/ReadVariableOp^Read_56/DisableCopyOnRead^Read_56/ReadVariableOp^Read_57/DisableCopyOnRead^Read_57/ReadVariableOp^Read_58/DisableCopyOnRead^Read_58/ReadVariableOp^Read_59/DisableCopyOnRead^Read_59/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_60/DisableCopyOnRead^Read_60/ReadVariableOp^Read_61/DisableCopyOnRead^Read_61/ReadVariableOp^Read_62/DisableCopyOnRead^Read_62/ReadVariableOp^Read_63/DisableCopyOnRead^Read_63/ReadVariableOp^Read_64/DisableCopyOnRead^Read_64/ReadVariableOp^Read_65/DisableCopyOnRead^Read_65/ReadVariableOp^Read_66/DisableCopyOnRead^Read_66/ReadVariableOp^Read_67/DisableCopyOnRead^Read_67/ReadVariableOp^Read_68/DisableCopyOnRead^Read_68/ReadVariableOp^Read_69/DisableCopyOnRead^Read_69/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_70/DisableCopyOnRead^Read_70/ReadVariableOp^Read_71/DisableCopyOnRead^Read_71/ReadVariableOp^Read_72/DisableCopyOnRead^Read_72/ReadVariableOp^Read_73/DisableCopyOnRead^Read_73/ReadVariableOp^Read_74/DisableCopyOnRead^Read_74/ReadVariableOp^Read_75/DisableCopyOnRead^Read_75/ReadVariableOp^Read_76/DisableCopyOnRead^Read_76/ReadVariableOp^Read_77/DisableCopyOnRead^Read_77/ReadVariableOp^Read_78/DisableCopyOnRead^Read_78/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "%
identity_159Identity_159:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp26
Read_34/DisableCopyOnReadRead_34/DisableCopyOnRead20
Read_34/ReadVariableOpRead_34/ReadVariableOp26
Read_35/DisableCopyOnReadRead_35/DisableCopyOnRead20
Read_35/ReadVariableOpRead_35/ReadVariableOp26
Read_36/DisableCopyOnReadRead_36/DisableCopyOnRead20
Read_36/ReadVariableOpRead_36/ReadVariableOp26
Read_37/DisableCopyOnReadRead_37/DisableCopyOnRead20
Read_37/ReadVariableOpRead_37/ReadVariableOp26
Read_38/DisableCopyOnReadRead_38/DisableCopyOnRead20
Read_38/ReadVariableOpRead_38/ReadVariableOp26
Read_39/DisableCopyOnReadRead_39/DisableCopyOnRead20
Read_39/ReadVariableOpRead_39/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp26
Read_40/DisableCopyOnReadRead_40/DisableCopyOnRead20
Read_40/ReadVariableOpRead_40/ReadVariableOp26
Read_41/DisableCopyOnReadRead_41/DisableCopyOnRead20
Read_41/ReadVariableOpRead_41/ReadVariableOp26
Read_42/DisableCopyOnReadRead_42/DisableCopyOnRead20
Read_42/ReadVariableOpRead_42/ReadVariableOp26
Read_43/DisableCopyOnReadRead_43/DisableCopyOnRead20
Read_43/ReadVariableOpRead_43/ReadVariableOp26
Read_44/DisableCopyOnReadRead_44/DisableCopyOnRead20
Read_44/ReadVariableOpRead_44/ReadVariableOp26
Read_45/DisableCopyOnReadRead_45/DisableCopyOnRead20
Read_45/ReadVariableOpRead_45/ReadVariableOp26
Read_46/DisableCopyOnReadRead_46/DisableCopyOnRead20
Read_46/ReadVariableOpRead_46/ReadVariableOp26
Read_47/DisableCopyOnReadRead_47/DisableCopyOnRead20
Read_47/ReadVariableOpRead_47/ReadVariableOp26
Read_48/DisableCopyOnReadRead_48/DisableCopyOnRead20
Read_48/ReadVariableOpRead_48/ReadVariableOp26
Read_49/DisableCopyOnReadRead_49/DisableCopyOnRead20
Read_49/ReadVariableOpRead_49/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp26
Read_50/DisableCopyOnReadRead_50/DisableCopyOnRead20
Read_50/ReadVariableOpRead_50/ReadVariableOp26
Read_51/DisableCopyOnReadRead_51/DisableCopyOnRead20
Read_51/ReadVariableOpRead_51/ReadVariableOp26
Read_52/DisableCopyOnReadRead_52/DisableCopyOnRead20
Read_52/ReadVariableOpRead_52/ReadVariableOp26
Read_53/DisableCopyOnReadRead_53/DisableCopyOnRead20
Read_53/ReadVariableOpRead_53/ReadVariableOp26
Read_54/DisableCopyOnReadRead_54/DisableCopyOnRead20
Read_54/ReadVariableOpRead_54/ReadVariableOp26
Read_55/DisableCopyOnReadRead_55/DisableCopyOnRead20
Read_55/ReadVariableOpRead_55/ReadVariableOp26
Read_56/DisableCopyOnReadRead_56/DisableCopyOnRead20
Read_56/ReadVariableOpRead_56/ReadVariableOp26
Read_57/DisableCopyOnReadRead_57/DisableCopyOnRead20
Read_57/ReadVariableOpRead_57/ReadVariableOp26
Read_58/DisableCopyOnReadRead_58/DisableCopyOnRead20
Read_58/ReadVariableOpRead_58/ReadVariableOp26
Read_59/DisableCopyOnReadRead_59/DisableCopyOnRead20
Read_59/ReadVariableOpRead_59/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp26
Read_60/DisableCopyOnReadRead_60/DisableCopyOnRead20
Read_60/ReadVariableOpRead_60/ReadVariableOp26
Read_61/DisableCopyOnReadRead_61/DisableCopyOnRead20
Read_61/ReadVariableOpRead_61/ReadVariableOp26
Read_62/DisableCopyOnReadRead_62/DisableCopyOnRead20
Read_62/ReadVariableOpRead_62/ReadVariableOp26
Read_63/DisableCopyOnReadRead_63/DisableCopyOnRead20
Read_63/ReadVariableOpRead_63/ReadVariableOp26
Read_64/DisableCopyOnReadRead_64/DisableCopyOnRead20
Read_64/ReadVariableOpRead_64/ReadVariableOp26
Read_65/DisableCopyOnReadRead_65/DisableCopyOnRead20
Read_65/ReadVariableOpRead_65/ReadVariableOp26
Read_66/DisableCopyOnReadRead_66/DisableCopyOnRead20
Read_66/ReadVariableOpRead_66/ReadVariableOp26
Read_67/DisableCopyOnReadRead_67/DisableCopyOnRead20
Read_67/ReadVariableOpRead_67/ReadVariableOp26
Read_68/DisableCopyOnReadRead_68/DisableCopyOnRead20
Read_68/ReadVariableOpRead_68/ReadVariableOp26
Read_69/DisableCopyOnReadRead_69/DisableCopyOnRead20
Read_69/ReadVariableOpRead_69/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp26
Read_70/DisableCopyOnReadRead_70/DisableCopyOnRead20
Read_70/ReadVariableOpRead_70/ReadVariableOp26
Read_71/DisableCopyOnReadRead_71/DisableCopyOnRead20
Read_71/ReadVariableOpRead_71/ReadVariableOp26
Read_72/DisableCopyOnReadRead_72/DisableCopyOnRead20
Read_72/ReadVariableOpRead_72/ReadVariableOp26
Read_73/DisableCopyOnReadRead_73/DisableCopyOnRead20
Read_73/ReadVariableOpRead_73/ReadVariableOp26
Read_74/DisableCopyOnReadRead_74/DisableCopyOnRead20
Read_74/ReadVariableOpRead_74/ReadVariableOp26
Read_75/DisableCopyOnReadRead_75/DisableCopyOnRead20
Read_75/ReadVariableOpRead_75/ReadVariableOp26
Read_76/DisableCopyOnReadRead_76/DisableCopyOnRead20
Read_76/ReadVariableOpRead_76/ReadVariableOp26
Read_77/DisableCopyOnReadRead_77/DisableCopyOnRead20
Read_77/ReadVariableOpRead_77/ReadVariableOp26
Read_78/DisableCopyOnReadRead_78/DisableCopyOnRead20
Read_78/ReadVariableOpRead_78/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:?P;

_output_shapes
: 
!
_user_specified_name	Const_2:<O8
6
_user_specified_namebatchnorm_3_/moving_variance:8N4
2
_user_specified_namebatchnorm_3_/moving_mean:8M4
2
_user_specified_namebatchnorm_1_/moving_mean:<L8
6
_user_specified_namebatchnorm_4_/moving_variance:8K4
2
_user_specified_namebatchnorm_5_/moving_mean:8J4
2
_user_specified_namebatchnorm_4_/moving_mean:8I4
2
_user_specified_namebatchnorm_2_/moving_mean:<H8
6
_user_specified_namebatchnorm_1_/moving_variance:<G8
6
_user_specified_namebatchnorm_2_/moving_variance:<F8
6
_user_specified_namebatchnorm_5_/moving_variance:,E(
&
_user_specified_nameconv_4_/bias:,D(
&
_user_specified_nameconv_2_/bias:2C.
,
_user_specified_namebatchnorm_1_/gamma:1B-
+
_user_specified_namebatchnorm_5_/beta:1A-
+
_user_specified_namebatchnorm_4_/beta:1@-
+
_user_specified_namebatchnorm_2_/beta:*?&
$
_user_specified_name
fc_2_/bias:2>.
,
_user_specified_namebatchnorm_3_/gamma:*=&
$
_user_specified_name
fc_1_/bias:4<0
.
_user_specified_namelstm_/lstm_cell/bias:,;(
&
_user_specified_namefc_2_/kernel:@:<
:
_user_specified_name" lstm_/lstm_cell/recurrent_kernel:29.
,
_user_specified_namebatchnorm_5_/gamma:28.
,
_user_specified_namebatchnorm_4_/gamma:27.
,
_user_specified_namebatchnorm_2_/gamma:,6(
&
_user_specified_nameconv_1_/bias:,5(
&
_user_specified_namefc_1_/kernel:642
0
_user_specified_namelstm_/lstm_cell/kernel:.3*
(
_user_specified_nameconv_3_/kernel:.2*
(
_user_specified_nameconv_5_/kernel:,1(
&
_user_specified_nameconv_3_/bias:.0*
(
_user_specified_nameconv_1_/kernel:,/(
&
_user_specified_nameconv_5_/bias:..*
(
_user_specified_nameconv_4_/kernel:1--
+
_user_specified_namebatchnorm_3_/beta:.,*
(
_user_specified_nameconv_2_/kernel:1+-
+
_user_specified_namebatchnorm_1_/beta:,*(
&
_user_specified_namefc_2_/bias_1:.)*
(
_user_specified_namefc_2_/kernel_1:4(0
.
_user_specified_nameseed_generator_state:,'(
&
_user_specified_namefc_1_/bias_1:.&*
(
_user_specified_namefc_1_/kernel_1:6%2
0
_user_specified_nameseed_generator_state_1:6$2
0
_user_specified_namelstm_/lstm_cell/bias_1:B#>
<
_user_specified_name$"lstm_/lstm_cell/recurrent_kernel_1:8"4
2
_user_specified_namelstm_/lstm_cell/kernel_1:>!:
8
_user_specified_name batchnorm_5_/moving_variance_1:: 6
4
_user_specified_namebatchnorm_5_/moving_mean_1:3/
-
_user_specified_namebatchnorm_5_/beta_1:40
.
_user_specified_namebatchnorm_5_/gamma_1:.*
(
_user_specified_nameconv_5_/bias_1:0,
*
_user_specified_nameconv_5_/kernel_1:>:
8
_user_specified_name batchnorm_4_/moving_variance_1::6
4
_user_specified_namebatchnorm_4_/moving_mean_1:3/
-
_user_specified_namebatchnorm_4_/beta_1:40
.
_user_specified_namebatchnorm_4_/gamma_1:.*
(
_user_specified_nameconv_4_/bias_1:0,
*
_user_specified_nameconv_4_/kernel_1:>:
8
_user_specified_name batchnorm_3_/moving_variance_1::6
4
_user_specified_namebatchnorm_3_/moving_mean_1:3/
-
_user_specified_namebatchnorm_3_/beta_1:40
.
_user_specified_namebatchnorm_3_/gamma_1:.*
(
_user_specified_nameconv_3_/bias_1:0,
*
_user_specified_nameconv_3_/kernel_1:>:
8
_user_specified_name batchnorm_2_/moving_variance_1::6
4
_user_specified_namebatchnorm_2_/moving_mean_1:3/
-
_user_specified_namebatchnorm_2_/beta_1:40
.
_user_specified_namebatchnorm_2_/gamma_1:.*
(
_user_specified_nameconv_2_/bias_1:0
,
*
_user_specified_nameconv_2_/kernel_1:>	:
8
_user_specified_name batchnorm_1_/moving_variance_1::6
4
_user_specified_namebatchnorm_1_/moving_mean_1:3/
-
_user_specified_namebatchnorm_1_/beta_1:40
.
_user_specified_namebatchnorm_1_/gamma_1:.*
(
_user_specified_nameconv_1_/bias_1:0,
*
_user_specified_nameconv_1_/kernel_1:1-
+
_user_specified_nameimageinput_/count:40
.
_user_specified_nameimageinput_/variance:0,
*
_user_specified_nameimageinput_/mean:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�Q
�
5__inference_functional_1_1_lstm__1_while_body_945_397J
Ffunctional_1_1_lstm__1_while_functional_1_1_lstm__1_while_loop_counter;
7functional_1_1_lstm__1_while_functional_1_1_lstm__1_max,
(functional_1_1_lstm__1_while_placeholder.
*functional_1_1_lstm__1_while_placeholder_1.
*functional_1_1_lstm__1_while_placeholder_2.
*functional_1_1_lstm__1_while_placeholder_3�
�functional_1_1_lstm__1_while_tensorarrayv2read_tensorlistgetitem_functional_1_1_lstm__1_tensorarrayunstack_tensorlistfromtensor_0[
Gfunctional_1_1_lstm__1_while_lstm_cell_1_cast_readvariableop_resource_0:
��]
Ifunctional_1_1_lstm__1_while_lstm_cell_1_cast_1_readvariableop_resource_0:
��W
Hfunctional_1_1_lstm__1_while_lstm_cell_1_add_1_readvariableop_resource_0:	�)
%functional_1_1_lstm__1_while_identity+
'functional_1_1_lstm__1_while_identity_1+
'functional_1_1_lstm__1_while_identity_2+
'functional_1_1_lstm__1_while_identity_3+
'functional_1_1_lstm__1_while_identity_4+
'functional_1_1_lstm__1_while_identity_5�
functional_1_1_lstm__1_while_tensorarrayv2read_tensorlistgetitem_functional_1_1_lstm__1_tensorarrayunstack_tensorlistfromtensorY
Efunctional_1_1_lstm__1_while_lstm_cell_1_cast_readvariableop_resource:
��[
Gfunctional_1_1_lstm__1_while_lstm_cell_1_cast_1_readvariableop_resource:
��U
Ffunctional_1_1_lstm__1_while_lstm_cell_1_add_1_readvariableop_resource:	���<functional_1_1/lstm__1/while/lstm_cell_1/Cast/ReadVariableOp�>functional_1_1/lstm__1/while/lstm_cell_1/Cast_1/ReadVariableOp�=functional_1_1/lstm__1/while/lstm_cell_1/add_1/ReadVariableOp�
Nfunctional_1_1/lstm__1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����0  �
@functional_1_1/lstm__1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem�functional_1_1_lstm__1_while_tensorarrayv2read_tensorlistgetitem_functional_1_1_lstm__1_tensorarrayunstack_tensorlistfromtensor_0(functional_1_1_lstm__1_while_placeholderWfunctional_1_1/lstm__1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:����������*
element_dtype0�
<functional_1_1/lstm__1/while/lstm_cell_1/Cast/ReadVariableOpReadVariableOpGfunctional_1_1_lstm__1_while_lstm_cell_1_cast_readvariableop_resource_0* 
_output_shapes
:
��*
dtype0�
/functional_1_1/lstm__1/while/lstm_cell_1/MatMulMatMulGfunctional_1_1/lstm__1/while/TensorArrayV2Read/TensorListGetItem:item:0Dfunctional_1_1/lstm__1/while/lstm_cell_1/Cast/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
>functional_1_1/lstm__1/while/lstm_cell_1/Cast_1/ReadVariableOpReadVariableOpIfunctional_1_1_lstm__1_while_lstm_cell_1_cast_1_readvariableop_resource_0* 
_output_shapes
:
��*
dtype0�
1functional_1_1/lstm__1/while/lstm_cell_1/MatMul_1MatMul*functional_1_1_lstm__1_while_placeholder_2Ffunctional_1_1/lstm__1/while/lstm_cell_1/Cast_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
,functional_1_1/lstm__1/while/lstm_cell_1/addAddV29functional_1_1/lstm__1/while/lstm_cell_1/MatMul:product:0;functional_1_1/lstm__1/while/lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
=functional_1_1/lstm__1/while/lstm_cell_1/add_1/ReadVariableOpReadVariableOpHfunctional_1_1_lstm__1_while_lstm_cell_1_add_1_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
.functional_1_1/lstm__1/while/lstm_cell_1/add_1AddV20functional_1_1/lstm__1/while/lstm_cell_1/add:z:0Efunctional_1_1/lstm__1/while/lstm_cell_1/add_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������z
8functional_1_1/lstm__1/while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
.functional_1_1/lstm__1/while/lstm_cell_1/splitSplitAfunctional_1_1/lstm__1/while/lstm_cell_1/split/split_dim:output:02functional_1_1/lstm__1/while/lstm_cell_1/add_1:z:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split�
0functional_1_1/lstm__1/while/lstm_cell_1/SigmoidSigmoid7functional_1_1/lstm__1/while/lstm_cell_1/split:output:0*
T0*(
_output_shapes
:�����������
2functional_1_1/lstm__1/while/lstm_cell_1/Sigmoid_1Sigmoid7functional_1_1/lstm__1/while/lstm_cell_1/split:output:1*
T0*(
_output_shapes
:�����������
,functional_1_1/lstm__1/while/lstm_cell_1/mulMul6functional_1_1/lstm__1/while/lstm_cell_1/Sigmoid_1:y:0*functional_1_1_lstm__1_while_placeholder_3*
T0*(
_output_shapes
:�����������
-functional_1_1/lstm__1/while/lstm_cell_1/TanhTanh7functional_1_1/lstm__1/while/lstm_cell_1/split:output:2*
T0*(
_output_shapes
:�����������
.functional_1_1/lstm__1/while/lstm_cell_1/mul_1Mul4functional_1_1/lstm__1/while/lstm_cell_1/Sigmoid:y:01functional_1_1/lstm__1/while/lstm_cell_1/Tanh:y:0*
T0*(
_output_shapes
:�����������
.functional_1_1/lstm__1/while/lstm_cell_1/add_2AddV20functional_1_1/lstm__1/while/lstm_cell_1/mul:z:02functional_1_1/lstm__1/while/lstm_cell_1/mul_1:z:0*
T0*(
_output_shapes
:�����������
2functional_1_1/lstm__1/while/lstm_cell_1/Sigmoid_2Sigmoid7functional_1_1/lstm__1/while/lstm_cell_1/split:output:3*
T0*(
_output_shapes
:�����������
/functional_1_1/lstm__1/while/lstm_cell_1/Tanh_1Tanh2functional_1_1/lstm__1/while/lstm_cell_1/add_2:z:0*
T0*(
_output_shapes
:�����������
.functional_1_1/lstm__1/while/lstm_cell_1/mul_2Mul6functional_1_1/lstm__1/while/lstm_cell_1/Sigmoid_2:y:03functional_1_1/lstm__1/while/lstm_cell_1/Tanh_1:y:0*
T0*(
_output_shapes
:�����������
Afunctional_1_1/lstm__1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem*functional_1_1_lstm__1_while_placeholder_1(functional_1_1_lstm__1_while_placeholder2functional_1_1/lstm__1/while/lstm_cell_1/mul_2:z:0*
_output_shapes
: *
element_dtype0:���d
"functional_1_1/lstm__1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
 functional_1_1/lstm__1/while/addAddV2(functional_1_1_lstm__1_while_placeholder+functional_1_1/lstm__1/while/add/y:output:0*
T0*
_output_shapes
: f
$functional_1_1/lstm__1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
"functional_1_1/lstm__1/while/add_1AddV2Ffunctional_1_1_lstm__1_while_functional_1_1_lstm__1_while_loop_counter-functional_1_1/lstm__1/while/add_1/y:output:0*
T0*
_output_shapes
: �
!functional_1_1/lstm__1/while/NoOpNoOp=^functional_1_1/lstm__1/while/lstm_cell_1/Cast/ReadVariableOp?^functional_1_1/lstm__1/while/lstm_cell_1/Cast_1/ReadVariableOp>^functional_1_1/lstm__1/while/lstm_cell_1/add_1/ReadVariableOp*
_output_shapes
 �
%functional_1_1/lstm__1/while/IdentityIdentity&functional_1_1/lstm__1/while/add_1:z:0"^functional_1_1/lstm__1/while/NoOp*
T0*
_output_shapes
: �
'functional_1_1/lstm__1/while/Identity_1Identity7functional_1_1_lstm__1_while_functional_1_1_lstm__1_max"^functional_1_1/lstm__1/while/NoOp*
T0*
_output_shapes
: �
'functional_1_1/lstm__1/while/Identity_2Identity$functional_1_1/lstm__1/while/add:z:0"^functional_1_1/lstm__1/while/NoOp*
T0*
_output_shapes
: �
'functional_1_1/lstm__1/while/Identity_3IdentityQfunctional_1_1/lstm__1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0"^functional_1_1/lstm__1/while/NoOp*
T0*
_output_shapes
: �
'functional_1_1/lstm__1/while/Identity_4Identity2functional_1_1/lstm__1/while/lstm_cell_1/mul_2:z:0"^functional_1_1/lstm__1/while/NoOp*
T0*(
_output_shapes
:�����������
'functional_1_1/lstm__1/while/Identity_5Identity2functional_1_1/lstm__1/while/lstm_cell_1/add_2:z:0"^functional_1_1/lstm__1/while/NoOp*
T0*(
_output_shapes
:����������"[
'functional_1_1_lstm__1_while_identity_10functional_1_1/lstm__1/while/Identity_1:output:0"[
'functional_1_1_lstm__1_while_identity_20functional_1_1/lstm__1/while/Identity_2:output:0"[
'functional_1_1_lstm__1_while_identity_30functional_1_1/lstm__1/while/Identity_3:output:0"[
'functional_1_1_lstm__1_while_identity_40functional_1_1/lstm__1/while/Identity_4:output:0"[
'functional_1_1_lstm__1_while_identity_50functional_1_1/lstm__1/while/Identity_5:output:0"W
%functional_1_1_lstm__1_while_identity.functional_1_1/lstm__1/while/Identity:output:0"�
Ffunctional_1_1_lstm__1_while_lstm_cell_1_add_1_readvariableop_resourceHfunctional_1_1_lstm__1_while_lstm_cell_1_add_1_readvariableop_resource_0"�
Gfunctional_1_1_lstm__1_while_lstm_cell_1_cast_1_readvariableop_resourceIfunctional_1_1_lstm__1_while_lstm_cell_1_cast_1_readvariableop_resource_0"�
Efunctional_1_1_lstm__1_while_lstm_cell_1_cast_readvariableop_resourceGfunctional_1_1_lstm__1_while_lstm_cell_1_cast_readvariableop_resource_0"�
functional_1_1_lstm__1_while_tensorarrayv2read_tensorlistgetitem_functional_1_1_lstm__1_tensorarrayunstack_tensorlistfromtensor�functional_1_1_lstm__1_while_tensorarrayv2read_tensorlistgetitem_functional_1_1_lstm__1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :����������:����������: : : : 2|
<functional_1_1/lstm__1/while/lstm_cell_1/Cast/ReadVariableOp<functional_1_1/lstm__1/while/lstm_cell_1/Cast/ReadVariableOp2�
>functional_1_1/lstm__1/while/lstm_cell_1/Cast_1/ReadVariableOp>functional_1_1/lstm__1/while/lstm_cell_1/Cast_1/ReadVariableOp2~
=functional_1_1/lstm__1/while/lstm_cell_1/add_1/ReadVariableOp=functional_1_1/lstm__1/while/lstm_cell_1/add_1/ReadVariableOp:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:vr

_output_shapes
: 
X
_user_specified_name@>functional_1_1/lstm__1/TensorArrayUnstack/TensorListFromTensor:.*
(
_output_shapes
:����������:.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
: :RN

_output_shapes
: 
4
_user_specified_namefunctional_1_1/lstm__1/Max:a ]

_output_shapes
: 
C
_user_specified_name+)functional_1_1/lstm__1/while/loop_counter
�
�
5__inference_functional_1_1_lstm__1_while_cond_944_351J
Ffunctional_1_1_lstm__1_while_functional_1_1_lstm__1_while_loop_counter;
7functional_1_1_lstm__1_while_functional_1_1_lstm__1_max,
(functional_1_1_lstm__1_while_placeholder.
*functional_1_1_lstm__1_while_placeholder_1.
*functional_1_1_lstm__1_while_placeholder_2.
*functional_1_1_lstm__1_while_placeholder_3_
[functional_1_1_lstm__1_while_functional_1_1_lstm__1_while_cond_944___redundant_placeholder0_
[functional_1_1_lstm__1_while_functional_1_1_lstm__1_while_cond_944___redundant_placeholder1_
[functional_1_1_lstm__1_while_functional_1_1_lstm__1_while_cond_944___redundant_placeholder2_
[functional_1_1_lstm__1_while_functional_1_1_lstm__1_while_cond_944___redundant_placeholder3)
%functional_1_1_lstm__1_while_identity
e
#functional_1_1/lstm__1/while/Less/yConst*
_output_shapes
: *
dtype0*
value	B :�
!functional_1_1/lstm__1/while/LessLess(functional_1_1_lstm__1_while_placeholder,functional_1_1/lstm__1/while/Less/y:output:0*
T0*
_output_shapes
: �
#functional_1_1/lstm__1/while/Less_1LessFfunctional_1_1_lstm__1_while_functional_1_1_lstm__1_while_loop_counter7functional_1_1_lstm__1_while_functional_1_1_lstm__1_max*
T0*
_output_shapes
: �
'functional_1_1/lstm__1/while/LogicalAnd
LogicalAnd'functional_1_1/lstm__1/while/Less_1:z:0%functional_1_1/lstm__1/while/Less:z:0*
_output_shapes
: 
%functional_1_1/lstm__1/while/IdentityIdentity+functional_1_1/lstm__1/while/LogicalAnd:z:0*
T0
*
_output_shapes
: "W
%functional_1_1_lstm__1_while_identity.functional_1_1/lstm__1/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :����������:����������:::::

_output_shapes
::.*
(
_output_shapes
:����������:.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
: :RN

_output_shapes
: 
4
_user_specified_namefunctional_1_1/lstm__1/Max:a ]

_output_shapes
: 
C
_user_specified_name+)functional_1_1/lstm__1/while/loop_counter
��
�	
__inference___call___1913	
input
unknown
	unknown_0#
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:$

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:$

unknown_19: 

unknown_20: 

unknown_21: 

unknown_22: 

unknown_23: 

unknown_24: $

unknown_25: P

unknown_26:P

unknown_27:P

unknown_28:P

unknown_29:P

unknown_30:P

unknown_31:
��

unknown_32:
��

unknown_33:	�

unknown_34:	�

unknown_35:

unknown_36:

unknown_37:
identity	

identity_1

identity_2��StatefulPartitionedCall3
ReadFileReadFileinput*
_output_shapes
: ~
	DecodeWav	DecodeWavReadFile:contents:0*!
_output_shapes
:	�}: *
desired_channels*
desired_samples�}k
SqueezeSqueezeDecodeWav:audio:0*
T0*
_output_shapes	
:�}*
squeeze_dims

���������L
	Squeeze_1SqueezeSqueeze:output:0*
T0*
_output_shapes	
:�}P
ShapeConst*
_output_shapes
:*
dtype0*
valueB:�}]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskH
sub/xConst*
_output_shapes
: *
dtype0*
value
B :�}S
subSubsub/x:output:0strided_slice:output:0*
T0*
_output_shapes
: K
	Maximum/xConst*
_output_shapes
: *
dtype0*
value	B : P
MaximumMaximumMaximum/x:output:0sub:z:0*
T0*
_output_shapes
: R
Pad/paddings/0/0Const*
_output_shapes
: *
dtype0*
value	B : l
Pad/paddings/0PackPad/paddings/0/0:output:0Maximum:z:0*
N*
T0*
_output_shapes
:_
Pad/paddingsPackPad/paddings/0:output:0*
N*
T0*
_output_shapes

:[
PadPadSqueeze_1:output:0Pad/paddings:output:0*
T0*
_output_shapes	
:�}Q
l2_normalize/SquareSquarePad:output:0*
T0*
_output_shapes	
:�}\
l2_normalize/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
l2_normalize/SumSuml2_normalize/Square:y:0l2_normalize/Const:output:0*
T0*
_output_shapes
:*
	keep_dims([
l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
l2_normalize/MaximumMaximuml2_normalize/Sum:output:0l2_normalize/Maximum/y:output:0*
T0*
_output_shapes
:Z
l2_normalize/RsqrtRsqrtl2_normalize/Maximum:z:0*
T0*
_output_shapes
:_
l2_normalizeMulPad:output:0l2_normalize/Rsqrt:y:0*
T0*
_output_shapes	
:�}T
stft/frame_lengthConst*
_output_shapes
: *
dtype0*
value
B :�R
stft/frame_stepConst*
_output_shapes
: *
dtype0*
value
B :�R
stft/fft_lengthConst*
_output_shapes
: *
dtype0*
value
B :�Z
stft/frame/axisConst*
_output_shapes
: *
dtype0*
valueB :
���������[
stft/frame/ShapeConst*
_output_shapes
:*
dtype0*
valueB:�}X
stft/frame/Size/ConstConst*
_output_shapes
: *
dtype0*
valueB Q
stft/frame/SizeConst*
_output_shapes
: *
dtype0*
value	B : Z
stft/frame/Size_1/ConstConst*
_output_shapes
: *
dtype0*
valueB S
stft/frame/Size_1Const*
_output_shapes
: *
dtype0*
value	B : R
stft/frame/ConstConst*
_output_shapes
: *
dtype0*
value	B : S
stft/frame/sub/xConst*
_output_shapes
: *
dtype0*
value
B :�}m
stft/frame/subSubstft/frame/sub/x:output:0stft/frame_length:output:0*
T0*
_output_shapes
: n
stft/frame/floordivFloorDivstft/frame/sub:z:0stft/frame_step:output:0*
T0*
_output_shapes
: R
stft/frame/add/xConst*
_output_shapes
: *
dtype0*
value	B :l
stft/frame/addAddV2stft/frame/add/x:output:0stft/frame/floordiv:z:0*
T0*
_output_shapes
: m
stft/frame/MaximumMaximumstft/frame/Const:output:0stft/frame/add:z:0*
T0*
_output_shapes
: V
stft/frame/gcd/ConstConst*
_output_shapes
: *
dtype0*
value	B :PY
stft/frame/floordiv_1/yConst*
_output_shapes
: *
dtype0*
value	B :P�
stft/frame/floordiv_1FloorDivstft/frame_length:output:0 stft/frame/floordiv_1/y:output:0*
T0*
_output_shapes
: Y
stft/frame/floordiv_2/yConst*
_output_shapes
: *
dtype0*
value	B :P~
stft/frame/floordiv_2FloorDivstft/frame_step:output:0 stft/frame/floordiv_2/y:output:0*
T0*
_output_shapes
: ]
stft/frame/concat/values_0Const*
_output_shapes
: *
dtype0*
valueB e
stft/frame/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:�}]
stft/frame/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB X
stft/frame/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
stft/frame/concatConcatV2#stft/frame/concat/values_0:output:0#stft/frame/concat/values_1:output:0#stft/frame/concat/values_2:output:0stft/frame/concat/axis:output:0*
N*
T0*
_output_shapes
:_
stft/frame/concat_1/values_0Const*
_output_shapes
: *
dtype0*
valueB m
stft/frame/concat_1/values_1Const*
_output_shapes
:*
dtype0*
valueB"�   P   _
stft/frame/concat_1/values_2Const*
_output_shapes
: *
dtype0*
valueB Z
stft/frame/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
stft/frame/concat_1ConcatV2%stft/frame/concat_1/values_0:output:0%stft/frame/concat_1/values_1:output:0%stft/frame/concat_1/values_2:output:0!stft/frame/concat_1/axis:output:0*
N*
T0*
_output_shapes
:g
stft/frame/zeros_like/tensorConst*
_output_shapes
:*
dtype0*
valueB:�}_
stft/frame/zeros_likeConst*
_output_shapes
:*
dtype0*
valueB: t
*stft/frame/ones_like/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:\
stft/frame/ones_like/ConstConst*
_output_shapes
: *
dtype0*
value	B :�
stft/frame/ones_likeFill3stft/frame/ones_like/Shape/shape_as_tensor:output:0#stft/frame/ones_like/Const:output:0*
T0*
_output_shapes
:�
stft/frame/StridedSliceStridedSlicel2_normalize:z:0stft/frame/zeros_like:output:0stft/frame/concat:output:0stft/frame/ones_like:output:0*
Index0*
T0*
_output_shapes	
:�}�
stft/frame/ReshapeReshape stft/frame/StridedSlice:output:0stft/frame/concat_1:output:0*
T0*
_output_shapes
:	�PX
stft/frame/range/startConst*
_output_shapes
: *
dtype0*
value	B : X
stft/frame/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
stft/frame/rangeRangestft/frame/range/start:output:0stft/frame/Maximum:z:0stft/frame/range/delta:output:0*
_output_shapes
:bp
stft/frame/mulMulstft/frame/range:output:0stft/frame/floordiv_2:z:0*
T0*
_output_shapes
:b^
stft/frame/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :�
stft/frame/Reshape_1/shapePackstft/frame/Maximum:z:0%stft/frame/Reshape_1/shape/1:output:0*
N*
T0*
_output_shapes
:�
stft/frame/Reshape_1Reshapestft/frame/mul:z:0#stft/frame/Reshape_1/shape:output:0*
T0*
_output_shapes

:bZ
stft/frame/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : Z
stft/frame/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
stft/frame/range_1Range!stft/frame/range_1/start:output:0stft/frame/floordiv_1:z:0!stft/frame/range_1/delta:output:0*
_output_shapes
:^
stft/frame/Reshape_2/shape/0Const*
_output_shapes
: *
dtype0*
value	B :�
stft/frame/Reshape_2/shapePack%stft/frame/Reshape_2/shape/0:output:0stft/frame/floordiv_1:z:0*
N*
T0*
_output_shapes
:�
stft/frame/Reshape_2Reshapestft/frame/range_1:output:0#stft/frame/Reshape_2/shape:output:0*
T0*
_output_shapes

:�
stft/frame/add_1AddV2stft/frame/Reshape_1:output:0stft/frame/Reshape_2:output:0*
T0*
_output_shapes

:bU
stft/frame/Const_1Const*
_output_shapes
: *
dtype0*
valueB U
stft/frame/Const_2Const*
_output_shapes
: *
dtype0*
valueB {
stft/frame/packedPackstft/frame/Maximum:z:0stft/frame_length:output:0*
N*
T0*
_output_shapes
:Z
stft/frame/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
stft/frame/GatherV2GatherV2stft/frame/Reshape:output:0stft/frame/add_1:z:0!stft/frame/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*"
_output_shapes
:bPZ
stft/frame/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
stft/frame/concat_2ConcatV2stft/frame/Const_1:output:0stft/frame/packed:output:0stft/frame/Const_2:output:0!stft/frame/concat_2/axis:output:0*
N*
T0*
_output_shapes
:�
stft/frame/Reshape_3Reshapestft/frame/GatherV2:output:0stft/frame/concat_2:output:0*
T0*
_output_shapes
:	b�[
stft/hann_window/periodicConst*
_output_shapes
: *
dtype0
*
value	B
 Zq
stft/hann_window/CastCast"stft/hann_window/periodic:output:0*

DstT0*

SrcT0
*
_output_shapes
: X
stft/hann_window/mod/yConst*
_output_shapes
: *
dtype0*
value	B :~
stft/hann_window/modFloorModstft/frame_length:output:0stft/hann_window/mod/y:output:0*
T0*
_output_shapes
: X
stft/hann_window/sub/xConst*
_output_shapes
: *
dtype0*
value	B :w
stft/hann_window/subSubstft/hann_window/sub/x:output:0stft/hann_window/mod:z:0*
T0*
_output_shapes
: q
stft/hann_window/mulMulstft/hann_window/Cast:y:0stft/hann_window/sub:z:0*
T0*
_output_shapes
: t
stft/hann_window/addAddV2stft/frame_length:output:0stft/hann_window/mul:z:0*
T0*
_output_shapes
: Z
stft/hann_window/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :{
stft/hann_window/sub_1Substft/hann_window/add:z:0!stft/hann_window/sub_1/y:output:0*
T0*
_output_shapes
: k
stft/hann_window/Cast_1Caststft/hann_window/sub_1:z:0*

DstT0*

SrcT0*
_output_shapes
: ^
stft/hann_window/range/startConst*
_output_shapes
: *
dtype0*
value	B : ^
stft/hann_window/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
stft/hann_window/rangeRange%stft/hann_window/range/start:output:0stft/frame_length:output:0%stft/hann_window/range/delta:output:0*
_output_shapes	
:�u
stft/hann_window/Cast_2Caststft/hann_window/range:output:0*

DstT0*

SrcT0*
_output_shapes	
:�[
stft/hann_window/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *��@�
stft/hann_window/mul_1Mulstft/hann_window/Const:output:0stft/hann_window/Cast_2:y:0*
T0*
_output_shapes	
:��
stft/hann_window/truedivRealDivstft/hann_window/mul_1:z:0stft/hann_window/Cast_1:y:0*
T0*
_output_shapes	
:�_
stft/hann_window/CosCosstft/hann_window/truediv:z:0*
T0*
_output_shapes	
:�]
stft/hann_window/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
stft/hann_window/mul_2Mul!stft/hann_window/mul_2/x:output:0stft/hann_window/Cos:y:0*
T0*
_output_shapes	
:�]
stft/hann_window/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
stft/hann_window/sub_2Sub!stft/hann_window/sub_2/x:output:0stft/hann_window/mul_2:z:0*
T0*
_output_shapes	
:�t
stft/mulMulstft/frame/Reshape_3:output:0stft/hann_window/sub_2:z:0*
T0*
_output_shapes
:	b�`
stft/rfft/packedPackstft/fft_length:output:0*
N*
T0*
_output_shapes
:w
stft/rfft/Pad/paddingsConst*
_output_shapes

:*
dtype0*)
value B"            p   m
stft/rfft/PadPadstft/mul:z:0stft/rfft/Pad/paddings:output:0*
T0*
_output_shapes
:	b�_
stft/rfft/fft_lengthConst*
_output_shapes
:*
dtype0*
valueB:�i
	stft/rfftRFFTstft/rfft/Pad:output:0stft/rfft/fft_length:output:0*
_output_shapes
:	b�F
Abs
ComplexAbsstft/rfft:output:0*
_output_shapes
:	b�X
Shape_1Const*
_output_shapes
:*
dtype0*
valueB"b     h
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
)linear_to_mel_weight_matrix/sample_rate/xConst*
_output_shapes
: *
dtype0*
value
B :�}�
'linear_to_mel_weight_matrix/sample_rateCast2linear_to_mel_weight_matrix/sample_rate/x:output:0*

DstT0*

SrcT0*
_output_shapes
: q
,linear_to_mel_weight_matrix/lower_edge_hertzConst*
_output_shapes
: *
dtype0*
valueB
 *  �Bq
,linear_to_mel_weight_matrix/upper_edge_hertzConst*
_output_shapes
: *
dtype0*
valueB
 * �mEf
!linear_to_mel_weight_matrix/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    j
%linear_to_mel_weight_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
#linear_to_mel_weight_matrix/truedivRealDiv+linear_to_mel_weight_matrix/sample_rate:y:0.linear_to_mel_weight_matrix/truediv/y:output:0*
T0*
_output_shapes
: {
)linear_to_mel_weight_matrix/linspace/CastCaststrided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: �
+linear_to_mel_weight_matrix/linspace/Cast_1Cast-linear_to_mel_weight_matrix/linspace/Cast:y:0*

DstT0*

SrcT0*
_output_shapes
: m
*linear_to_mel_weight_matrix/linspace/ShapeConst*
_output_shapes
: *
dtype0*
valueB o
,linear_to_mel_weight_matrix/linspace/Shape_1Const*
_output_shapes
: *
dtype0*
valueB �
2linear_to_mel_weight_matrix/linspace/BroadcastArgsBroadcastArgs3linear_to_mel_weight_matrix/linspace/Shape:output:05linear_to_mel_weight_matrix/linspace/Shape_1:output:0*
_output_shapes
: �
0linear_to_mel_weight_matrix/linspace/BroadcastToBroadcastTo*linear_to_mel_weight_matrix/Const:output:07linear_to_mel_weight_matrix/linspace/BroadcastArgs:r0:0*
T0*
_output_shapes
: �
2linear_to_mel_weight_matrix/linspace/BroadcastTo_1BroadcastTo'linear_to_mel_weight_matrix/truediv:z:07linear_to_mel_weight_matrix/linspace/BroadcastArgs:r0:0*
T0*
_output_shapes
: u
3linear_to_mel_weight_matrix/linspace/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : �
/linear_to_mel_weight_matrix/linspace/ExpandDims
ExpandDims9linear_to_mel_weight_matrix/linspace/BroadcastTo:output:0<linear_to_mel_weight_matrix/linspace/ExpandDims/dim:output:0*
T0*
_output_shapes
:w
5linear_to_mel_weight_matrix/linspace/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
1linear_to_mel_weight_matrix/linspace/ExpandDims_1
ExpandDims;linear_to_mel_weight_matrix/linspace/BroadcastTo_1:output:0>linear_to_mel_weight_matrix/linspace/ExpandDims_1/dim:output:0*
T0*
_output_shapes
:v
,linear_to_mel_weight_matrix/linspace/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:v
,linear_to_mel_weight_matrix/linspace/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:�
8linear_to_mel_weight_matrix/linspace/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
:linear_to_mel_weight_matrix/linspace/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
:linear_to_mel_weight_matrix/linspace/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
2linear_to_mel_weight_matrix/linspace/strided_sliceStridedSlice5linear_to_mel_weight_matrix/linspace/Shape_3:output:0Alinear_to_mel_weight_matrix/linspace/strided_slice/stack:output:0Clinear_to_mel_weight_matrix/linspace/strided_slice/stack_1:output:0Clinear_to_mel_weight_matrix/linspace/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
*linear_to_mel_weight_matrix/linspace/add/yConst*
_output_shapes
: *
dtype0*
value	B : �
(linear_to_mel_weight_matrix/linspace/addAddV2;linear_to_mel_weight_matrix/linspace/strided_slice:output:03linear_to_mel_weight_matrix/linspace/add/y:output:0*
T0*
_output_shapes
: y
7linear_to_mel_weight_matrix/linspace/SelectV2/conditionConst*
_output_shapes
: *
dtype0
*
value	B
 Zq
/linear_to_mel_weight_matrix/linspace/SelectV2/tConst*
_output_shapes
: *
dtype0*
value	B : �
-linear_to_mel_weight_matrix/linspace/SelectV2SelectV2@linear_to_mel_weight_matrix/linspace/SelectV2/condition:output:08linear_to_mel_weight_matrix/linspace/SelectV2/t:output:0,linear_to_mel_weight_matrix/linspace/add:z:0*
T0*
_output_shapes
: l
*linear_to_mel_weight_matrix/linspace/sub/yConst*
_output_shapes
: *
dtype0*
value	B :�
(linear_to_mel_weight_matrix/linspace/subSub-linear_to_mel_weight_matrix/linspace/Cast:y:03linear_to_mel_weight_matrix/linspace/sub/y:output:0*
T0*
_output_shapes
: p
.linear_to_mel_weight_matrix/linspace/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B : �
,linear_to_mel_weight_matrix/linspace/MaximumMaximum,linear_to_mel_weight_matrix/linspace/sub:z:07linear_to_mel_weight_matrix/linspace/Maximum/y:output:0*
T0*
_output_shapes
: n
,linear_to_mel_weight_matrix/linspace/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
*linear_to_mel_weight_matrix/linspace/sub_1Sub-linear_to_mel_weight_matrix/linspace/Cast:y:05linear_to_mel_weight_matrix/linspace/sub_1/y:output:0*
T0*
_output_shapes
: r
0linear_to_mel_weight_matrix/linspace/Maximum_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
.linear_to_mel_weight_matrix/linspace/Maximum_1Maximum.linear_to_mel_weight_matrix/linspace/sub_1:z:09linear_to_mel_weight_matrix/linspace/Maximum_1/y:output:0*
T0*
_output_shapes
: �
*linear_to_mel_weight_matrix/linspace/sub_2Sub:linear_to_mel_weight_matrix/linspace/ExpandDims_1:output:08linear_to_mel_weight_matrix/linspace/ExpandDims:output:0*
T0*
_output_shapes
:�
+linear_to_mel_weight_matrix/linspace/Cast_2Cast2linear_to_mel_weight_matrix/linspace/Maximum_1:z:0*

DstT0*

SrcT0*
_output_shapes
: �
,linear_to_mel_weight_matrix/linspace/truedivRealDiv.linear_to_mel_weight_matrix/linspace/sub_2:z:0/linear_to_mel_weight_matrix/linspace/Cast_2:y:0*
T0*
_output_shapes
:u
3linear_to_mel_weight_matrix/linspace/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : �
1linear_to_mel_weight_matrix/linspace/GreaterEqualGreaterEqual-linear_to_mel_weight_matrix/linspace/Cast:y:0<linear_to_mel_weight_matrix/linspace/GreaterEqual/y:output:0*
T0*
_output_shapes
: |
1linear_to_mel_weight_matrix/linspace/SelectV2_1/eConst*
_output_shapes
: *
dtype0*
valueB :
����������
/linear_to_mel_weight_matrix/linspace/SelectV2_1SelectV25linear_to_mel_weight_matrix/linspace/GreaterEqual:z:02linear_to_mel_weight_matrix/linspace/Maximum_1:z:0:linear_to_mel_weight_matrix/linspace/SelectV2_1/e:output:0*
T0*
_output_shapes
: r
0linear_to_mel_weight_matrix/linspace/range/startConst*
_output_shapes
: *
dtype0	*
value	B	 Rr
0linear_to_mel_weight_matrix/linspace/range/deltaConst*
_output_shapes
: *
dtype0	*
value	B	 R�
/linear_to_mel_weight_matrix/linspace/range/CastCast8linear_to_mel_weight_matrix/linspace/SelectV2_1:output:0*

DstT0	*

SrcT0*
_output_shapes
: �
*linear_to_mel_weight_matrix/linspace/rangeRange9linear_to_mel_weight_matrix/linspace/range/start:output:03linear_to_mel_weight_matrix/linspace/range/Cast:y:09linear_to_mel_weight_matrix/linspace/range/delta:output:0*

Tidx0	*
_output_shapes	
:��
+linear_to_mel_weight_matrix/linspace/Cast_3Cast3linear_to_mel_weight_matrix/linspace/range:output:0*

DstT0*

SrcT0	*
_output_shapes	
:�t
2linear_to_mel_weight_matrix/linspace/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : t
2linear_to_mel_weight_matrix/linspace/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
,linear_to_mel_weight_matrix/linspace/range_1Range;linear_to_mel_weight_matrix/linspace/range_1/start:output:0;linear_to_mel_weight_matrix/linspace/strided_slice:output:0;linear_to_mel_weight_matrix/linspace/range_1/delta:output:0*
_output_shapes
:�
*linear_to_mel_weight_matrix/linspace/EqualEqual6linear_to_mel_weight_matrix/linspace/SelectV2:output:05linear_to_mel_weight_matrix/linspace/range_1:output:0*
T0*
_output_shapes
:s
1linear_to_mel_weight_matrix/linspace/SelectV2_2/eConst*
_output_shapes
: *
dtype0*
value	B :�
/linear_to_mel_weight_matrix/linspace/SelectV2_2SelectV2.linear_to_mel_weight_matrix/linspace/Equal:z:00linear_to_mel_weight_matrix/linspace/Maximum:z:0:linear_to_mel_weight_matrix/linspace/SelectV2_2/e:output:0*
T0*
_output_shapes
:�
,linear_to_mel_weight_matrix/linspace/ReshapeReshape/linear_to_mel_weight_matrix/linspace/Cast_3:y:08linear_to_mel_weight_matrix/linspace/SelectV2_2:output:0*
T0*
_output_shapes	
:��
(linear_to_mel_weight_matrix/linspace/mulMul0linear_to_mel_weight_matrix/linspace/truediv:z:05linear_to_mel_weight_matrix/linspace/Reshape:output:0*
T0*
_output_shapes	
:��
*linear_to_mel_weight_matrix/linspace/add_1AddV28linear_to_mel_weight_matrix/linspace/ExpandDims:output:0,linear_to_mel_weight_matrix/linspace/mul:z:0*
T0*
_output_shapes	
:��
+linear_to_mel_weight_matrix/linspace/concatConcatV28linear_to_mel_weight_matrix/linspace/ExpandDims:output:0.linear_to_mel_weight_matrix/linspace/add_1:z:0:linear_to_mel_weight_matrix/linspace/ExpandDims_1:output:06linear_to_mel_weight_matrix/linspace/SelectV2:output:0*
N*
T0*
_output_shapes	
:�y
/linear_to_mel_weight_matrix/linspace/zeros_likeConst*
_output_shapes
:*
dtype0*
valueB: l
*linear_to_mel_weight_matrix/linspace/ConstConst*
_output_shapes
: *
dtype0*
value	B :�
:linear_to_mel_weight_matrix/linspace/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
<linear_to_mel_weight_matrix/linspace/strided_slice_1/stack_1Pack6linear_to_mel_weight_matrix/linspace/SelectV2:output:0*
N*
T0*
_output_shapes
:�
<linear_to_mel_weight_matrix/linspace/strided_slice_1/stack_2Pack3linear_to_mel_weight_matrix/linspace/Const:output:0*
N*
T0*
_output_shapes
:�
4linear_to_mel_weight_matrix/linspace/strided_slice_1StridedSlice5linear_to_mel_weight_matrix/linspace/Shape_2:output:0Clinear_to_mel_weight_matrix/linspace/strided_slice_1/stack:output:0Elinear_to_mel_weight_matrix/linspace/strided_slice_1/stack_1:output:0Elinear_to_mel_weight_matrix/linspace/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: ~
4linear_to_mel_weight_matrix/linspace/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:�
.linear_to_mel_weight_matrix/linspace/Reshape_1Reshape-linear_to_mel_weight_matrix/linspace/Cast:y:0=linear_to_mel_weight_matrix/linspace/Reshape_1/shape:output:0*
T0*
_output_shapes
:n
,linear_to_mel_weight_matrix/linspace/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :�
*linear_to_mel_weight_matrix/linspace/add_2AddV26linear_to_mel_weight_matrix/linspace/SelectV2:output:05linear_to_mel_weight_matrix/linspace/add_2/y:output:0*
T0*
_output_shapes
: n
,linear_to_mel_weight_matrix/linspace/Const_1Const*
_output_shapes
: *
dtype0*
value	B : n
,linear_to_mel_weight_matrix/linspace/Const_2Const*
_output_shapes
: *
dtype0*
value	B :�
:linear_to_mel_weight_matrix/linspace/strided_slice_2/stackPack.linear_to_mel_weight_matrix/linspace/add_2:z:0*
N*
T0*
_output_shapes
:�
<linear_to_mel_weight_matrix/linspace/strided_slice_2/stack_1Pack5linear_to_mel_weight_matrix/linspace/Const_1:output:0*
N*
T0*
_output_shapes
:�
<linear_to_mel_weight_matrix/linspace/strided_slice_2/stack_2Pack5linear_to_mel_weight_matrix/linspace/Const_2:output:0*
N*
T0*
_output_shapes
:�
4linear_to_mel_weight_matrix/linspace/strided_slice_2StridedSlice5linear_to_mel_weight_matrix/linspace/Shape_2:output:0Clinear_to_mel_weight_matrix/linspace/strided_slice_2/stack:output:0Elinear_to_mel_weight_matrix/linspace/strided_slice_2/stack_1:output:0Elinear_to_mel_weight_matrix/linspace/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_maskt
2linear_to_mel_weight_matrix/linspace/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
-linear_to_mel_weight_matrix/linspace/concat_1ConcatV2=linear_to_mel_weight_matrix/linspace/strided_slice_1:output:07linear_to_mel_weight_matrix/linspace/Reshape_1:output:0=linear_to_mel_weight_matrix/linspace/strided_slice_2:output:0;linear_to_mel_weight_matrix/linspace/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
*linear_to_mel_weight_matrix/linspace/SliceSlice4linear_to_mel_weight_matrix/linspace/concat:output:08linear_to_mel_weight_matrix/linspace/zeros_like:output:06linear_to_mel_weight_matrix/linspace/concat_1:output:0*
Index0*
T0*
_output_shapes	
:�y
/linear_to_mel_weight_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:{
1linear_to_mel_weight_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: {
1linear_to_mel_weight_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
)linear_to_mel_weight_matrix/strided_sliceStridedSlice3linear_to_mel_weight_matrix/linspace/Slice:output:08linear_to_mel_weight_matrix/strided_slice/stack:output:0:linear_to_mel_weight_matrix/strided_slice/stack_1:output:0:linear_to_mel_weight_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes	
:�*
end_maskw
2linear_to_mel_weight_matrix/hertz_to_mel/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  /D�
0linear_to_mel_weight_matrix/hertz_to_mel/truedivRealDiv2linear_to_mel_weight_matrix/strided_slice:output:0;linear_to_mel_weight_matrix/hertz_to_mel/truediv/y:output:0*
T0*
_output_shapes	
:�s
.linear_to_mel_weight_matrix/hertz_to_mel/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
,linear_to_mel_weight_matrix/hertz_to_mel/addAddV27linear_to_mel_weight_matrix/hertz_to_mel/add/x:output:04linear_to_mel_weight_matrix/hertz_to_mel/truediv:z:0*
T0*
_output_shapes	
:��
,linear_to_mel_weight_matrix/hertz_to_mel/LogLog0linear_to_mel_weight_matrix/hertz_to_mel/add:z:0*
T0*
_output_shapes	
:�s
.linear_to_mel_weight_matrix/hertz_to_mel/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 * ��D�
,linear_to_mel_weight_matrix/hertz_to_mel/mulMul7linear_to_mel_weight_matrix/hertz_to_mel/mul/x:output:00linear_to_mel_weight_matrix/hertz_to_mel/Log:y:0*
T0*
_output_shapes	
:�l
*linear_to_mel_weight_matrix/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
&linear_to_mel_weight_matrix/ExpandDims
ExpandDims0linear_to_mel_weight_matrix/hertz_to_mel/mul:z:03linear_to_mel_weight_matrix/ExpandDims/dim:output:0*
T0*
_output_shapes
:	�y
4linear_to_mel_weight_matrix/hertz_to_mel_1/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  /D�
2linear_to_mel_weight_matrix/hertz_to_mel_1/truedivRealDiv5linear_to_mel_weight_matrix/lower_edge_hertz:output:0=linear_to_mel_weight_matrix/hertz_to_mel_1/truediv/y:output:0*
T0*
_output_shapes
: u
0linear_to_mel_weight_matrix/hertz_to_mel_1/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
.linear_to_mel_weight_matrix/hertz_to_mel_1/addAddV29linear_to_mel_weight_matrix/hertz_to_mel_1/add/x:output:06linear_to_mel_weight_matrix/hertz_to_mel_1/truediv:z:0*
T0*
_output_shapes
: �
.linear_to_mel_weight_matrix/hertz_to_mel_1/LogLog2linear_to_mel_weight_matrix/hertz_to_mel_1/add:z:0*
T0*
_output_shapes
: u
0linear_to_mel_weight_matrix/hertz_to_mel_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 * ��D�
.linear_to_mel_weight_matrix/hertz_to_mel_1/mulMul9linear_to_mel_weight_matrix/hertz_to_mel_1/mul/x:output:02linear_to_mel_weight_matrix/hertz_to_mel_1/Log:y:0*
T0*
_output_shapes
: y
4linear_to_mel_weight_matrix/hertz_to_mel_2/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  /D�
2linear_to_mel_weight_matrix/hertz_to_mel_2/truedivRealDiv5linear_to_mel_weight_matrix/upper_edge_hertz:output:0=linear_to_mel_weight_matrix/hertz_to_mel_2/truediv/y:output:0*
T0*
_output_shapes
: u
0linear_to_mel_weight_matrix/hertz_to_mel_2/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
.linear_to_mel_weight_matrix/hertz_to_mel_2/addAddV29linear_to_mel_weight_matrix/hertz_to_mel_2/add/x:output:06linear_to_mel_weight_matrix/hertz_to_mel_2/truediv:z:0*
T0*
_output_shapes
: �
.linear_to_mel_weight_matrix/hertz_to_mel_2/LogLog2linear_to_mel_weight_matrix/hertz_to_mel_2/add:z:0*
T0*
_output_shapes
: u
0linear_to_mel_weight_matrix/hertz_to_mel_2/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 * ��D�
.linear_to_mel_weight_matrix/hertz_to_mel_2/mulMul9linear_to_mel_weight_matrix/hertz_to_mel_2/mul/x:output:02linear_to_mel_weight_matrix/hertz_to_mel_2/Log:y:0*
T0*
_output_shapes
: l
*linear_to_mel_weight_matrix/linspace_1/numConst*
_output_shapes
: *
dtype0*
value	B :4�
+linear_to_mel_weight_matrix/linspace_1/CastCast3linear_to_mel_weight_matrix/linspace_1/num:output:0*

DstT0*

SrcT0*
_output_shapes
: �
-linear_to_mel_weight_matrix/linspace_1/Cast_1Cast/linear_to_mel_weight_matrix/linspace_1/Cast:y:0*

DstT0*

SrcT0*
_output_shapes
: o
,linear_to_mel_weight_matrix/linspace_1/ShapeConst*
_output_shapes
: *
dtype0*
valueB q
.linear_to_mel_weight_matrix/linspace_1/Shape_1Const*
_output_shapes
: *
dtype0*
valueB �
4linear_to_mel_weight_matrix/linspace_1/BroadcastArgsBroadcastArgs5linear_to_mel_weight_matrix/linspace_1/Shape:output:07linear_to_mel_weight_matrix/linspace_1/Shape_1:output:0*
_output_shapes
: �
2linear_to_mel_weight_matrix/linspace_1/BroadcastToBroadcastTo2linear_to_mel_weight_matrix/hertz_to_mel_1/mul:z:09linear_to_mel_weight_matrix/linspace_1/BroadcastArgs:r0:0*
T0*
_output_shapes
: �
4linear_to_mel_weight_matrix/linspace_1/BroadcastTo_1BroadcastTo2linear_to_mel_weight_matrix/hertz_to_mel_2/mul:z:09linear_to_mel_weight_matrix/linspace_1/BroadcastArgs:r0:0*
T0*
_output_shapes
: w
5linear_to_mel_weight_matrix/linspace_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : �
1linear_to_mel_weight_matrix/linspace_1/ExpandDims
ExpandDims;linear_to_mel_weight_matrix/linspace_1/BroadcastTo:output:0>linear_to_mel_weight_matrix/linspace_1/ExpandDims/dim:output:0*
T0*
_output_shapes
:y
7linear_to_mel_weight_matrix/linspace_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
3linear_to_mel_weight_matrix/linspace_1/ExpandDims_1
ExpandDims=linear_to_mel_weight_matrix/linspace_1/BroadcastTo_1:output:0@linear_to_mel_weight_matrix/linspace_1/ExpandDims_1/dim:output:0*
T0*
_output_shapes
:x
.linear_to_mel_weight_matrix/linspace_1/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:x
.linear_to_mel_weight_matrix/linspace_1/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:�
:linear_to_mel_weight_matrix/linspace_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
<linear_to_mel_weight_matrix/linspace_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
<linear_to_mel_weight_matrix/linspace_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
4linear_to_mel_weight_matrix/linspace_1/strided_sliceStridedSlice7linear_to_mel_weight_matrix/linspace_1/Shape_3:output:0Clinear_to_mel_weight_matrix/linspace_1/strided_slice/stack:output:0Elinear_to_mel_weight_matrix/linspace_1/strided_slice/stack_1:output:0Elinear_to_mel_weight_matrix/linspace_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
,linear_to_mel_weight_matrix/linspace_1/add/yConst*
_output_shapes
: *
dtype0*
value	B : �
*linear_to_mel_weight_matrix/linspace_1/addAddV2=linear_to_mel_weight_matrix/linspace_1/strided_slice:output:05linear_to_mel_weight_matrix/linspace_1/add/y:output:0*
T0*
_output_shapes
: {
9linear_to_mel_weight_matrix/linspace_1/SelectV2/conditionConst*
_output_shapes
: *
dtype0
*
value	B
 Zs
1linear_to_mel_weight_matrix/linspace_1/SelectV2/tConst*
_output_shapes
: *
dtype0*
value	B : �
/linear_to_mel_weight_matrix/linspace_1/SelectV2SelectV2Blinear_to_mel_weight_matrix/linspace_1/SelectV2/condition:output:0:linear_to_mel_weight_matrix/linspace_1/SelectV2/t:output:0.linear_to_mel_weight_matrix/linspace_1/add:z:0*
T0*
_output_shapes
: n
,linear_to_mel_weight_matrix/linspace_1/sub/yConst*
_output_shapes
: *
dtype0*
value	B :�
*linear_to_mel_weight_matrix/linspace_1/subSub/linear_to_mel_weight_matrix/linspace_1/Cast:y:05linear_to_mel_weight_matrix/linspace_1/sub/y:output:0*
T0*
_output_shapes
: r
0linear_to_mel_weight_matrix/linspace_1/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B : �
.linear_to_mel_weight_matrix/linspace_1/MaximumMaximum.linear_to_mel_weight_matrix/linspace_1/sub:z:09linear_to_mel_weight_matrix/linspace_1/Maximum/y:output:0*
T0*
_output_shapes
: p
.linear_to_mel_weight_matrix/linspace_1/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
,linear_to_mel_weight_matrix/linspace_1/sub_1Sub/linear_to_mel_weight_matrix/linspace_1/Cast:y:07linear_to_mel_weight_matrix/linspace_1/sub_1/y:output:0*
T0*
_output_shapes
: t
2linear_to_mel_weight_matrix/linspace_1/Maximum_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
0linear_to_mel_weight_matrix/linspace_1/Maximum_1Maximum0linear_to_mel_weight_matrix/linspace_1/sub_1:z:0;linear_to_mel_weight_matrix/linspace_1/Maximum_1/y:output:0*
T0*
_output_shapes
: �
,linear_to_mel_weight_matrix/linspace_1/sub_2Sub<linear_to_mel_weight_matrix/linspace_1/ExpandDims_1:output:0:linear_to_mel_weight_matrix/linspace_1/ExpandDims:output:0*
T0*
_output_shapes
:�
-linear_to_mel_weight_matrix/linspace_1/Cast_2Cast4linear_to_mel_weight_matrix/linspace_1/Maximum_1:z:0*

DstT0*

SrcT0*
_output_shapes
: �
.linear_to_mel_weight_matrix/linspace_1/truedivRealDiv0linear_to_mel_weight_matrix/linspace_1/sub_2:z:01linear_to_mel_weight_matrix/linspace_1/Cast_2:y:0*
T0*
_output_shapes
:w
5linear_to_mel_weight_matrix/linspace_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : �
3linear_to_mel_weight_matrix/linspace_1/GreaterEqualGreaterEqual/linear_to_mel_weight_matrix/linspace_1/Cast:y:0>linear_to_mel_weight_matrix/linspace_1/GreaterEqual/y:output:0*
T0*
_output_shapes
: ~
3linear_to_mel_weight_matrix/linspace_1/SelectV2_1/eConst*
_output_shapes
: *
dtype0*
valueB :
����������
1linear_to_mel_weight_matrix/linspace_1/SelectV2_1SelectV27linear_to_mel_weight_matrix/linspace_1/GreaterEqual:z:04linear_to_mel_weight_matrix/linspace_1/Maximum_1:z:0<linear_to_mel_weight_matrix/linspace_1/SelectV2_1/e:output:0*
T0*
_output_shapes
: t
2linear_to_mel_weight_matrix/linspace_1/range/startConst*
_output_shapes
: *
dtype0	*
value	B	 Rt
2linear_to_mel_weight_matrix/linspace_1/range/deltaConst*
_output_shapes
: *
dtype0	*
value	B	 R�
1linear_to_mel_weight_matrix/linspace_1/range/CastCast:linear_to_mel_weight_matrix/linspace_1/SelectV2_1:output:0*

DstT0	*

SrcT0*
_output_shapes
: �
,linear_to_mel_weight_matrix/linspace_1/rangeRange;linear_to_mel_weight_matrix/linspace_1/range/start:output:05linear_to_mel_weight_matrix/linspace_1/range/Cast:y:0;linear_to_mel_weight_matrix/linspace_1/range/delta:output:0*

Tidx0	*
_output_shapes
:2�
-linear_to_mel_weight_matrix/linspace_1/Cast_3Cast5linear_to_mel_weight_matrix/linspace_1/range:output:0*

DstT0*

SrcT0	*
_output_shapes
:2v
4linear_to_mel_weight_matrix/linspace_1/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : v
4linear_to_mel_weight_matrix/linspace_1/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
.linear_to_mel_weight_matrix/linspace_1/range_1Range=linear_to_mel_weight_matrix/linspace_1/range_1/start:output:0=linear_to_mel_weight_matrix/linspace_1/strided_slice:output:0=linear_to_mel_weight_matrix/linspace_1/range_1/delta:output:0*
_output_shapes
:�
,linear_to_mel_weight_matrix/linspace_1/EqualEqual8linear_to_mel_weight_matrix/linspace_1/SelectV2:output:07linear_to_mel_weight_matrix/linspace_1/range_1:output:0*
T0*
_output_shapes
:u
3linear_to_mel_weight_matrix/linspace_1/SelectV2_2/eConst*
_output_shapes
: *
dtype0*
value	B :�
1linear_to_mel_weight_matrix/linspace_1/SelectV2_2SelectV20linear_to_mel_weight_matrix/linspace_1/Equal:z:02linear_to_mel_weight_matrix/linspace_1/Maximum:z:0<linear_to_mel_weight_matrix/linspace_1/SelectV2_2/e:output:0*
T0*
_output_shapes
:�
.linear_to_mel_weight_matrix/linspace_1/ReshapeReshape1linear_to_mel_weight_matrix/linspace_1/Cast_3:y:0:linear_to_mel_weight_matrix/linspace_1/SelectV2_2:output:0*
T0*
_output_shapes
:2�
*linear_to_mel_weight_matrix/linspace_1/mulMul2linear_to_mel_weight_matrix/linspace_1/truediv:z:07linear_to_mel_weight_matrix/linspace_1/Reshape:output:0*
T0*
_output_shapes
:2�
,linear_to_mel_weight_matrix/linspace_1/add_1AddV2:linear_to_mel_weight_matrix/linspace_1/ExpandDims:output:0.linear_to_mel_weight_matrix/linspace_1/mul:z:0*
T0*
_output_shapes
:2�
-linear_to_mel_weight_matrix/linspace_1/concatConcatV2:linear_to_mel_weight_matrix/linspace_1/ExpandDims:output:00linear_to_mel_weight_matrix/linspace_1/add_1:z:0<linear_to_mel_weight_matrix/linspace_1/ExpandDims_1:output:08linear_to_mel_weight_matrix/linspace_1/SelectV2:output:0*
N*
T0*
_output_shapes
:4{
1linear_to_mel_weight_matrix/linspace_1/zeros_likeConst*
_output_shapes
:*
dtype0*
valueB: n
,linear_to_mel_weight_matrix/linspace_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :�
<linear_to_mel_weight_matrix/linspace_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
>linear_to_mel_weight_matrix/linspace_1/strided_slice_1/stack_1Pack8linear_to_mel_weight_matrix/linspace_1/SelectV2:output:0*
N*
T0*
_output_shapes
:�
>linear_to_mel_weight_matrix/linspace_1/strided_slice_1/stack_2Pack5linear_to_mel_weight_matrix/linspace_1/Const:output:0*
N*
T0*
_output_shapes
:�
6linear_to_mel_weight_matrix/linspace_1/strided_slice_1StridedSlice7linear_to_mel_weight_matrix/linspace_1/Shape_2:output:0Elinear_to_mel_weight_matrix/linspace_1/strided_slice_1/stack:output:0Glinear_to_mel_weight_matrix/linspace_1/strided_slice_1/stack_1:output:0Glinear_to_mel_weight_matrix/linspace_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: �
6linear_to_mel_weight_matrix/linspace_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:�
0linear_to_mel_weight_matrix/linspace_1/Reshape_1Reshape/linear_to_mel_weight_matrix/linspace_1/Cast:y:0?linear_to_mel_weight_matrix/linspace_1/Reshape_1/shape:output:0*
T0*
_output_shapes
:p
.linear_to_mel_weight_matrix/linspace_1/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :�
,linear_to_mel_weight_matrix/linspace_1/add_2AddV28linear_to_mel_weight_matrix/linspace_1/SelectV2:output:07linear_to_mel_weight_matrix/linspace_1/add_2/y:output:0*
T0*
_output_shapes
: p
.linear_to_mel_weight_matrix/linspace_1/Const_1Const*
_output_shapes
: *
dtype0*
value	B : p
.linear_to_mel_weight_matrix/linspace_1/Const_2Const*
_output_shapes
: *
dtype0*
value	B :�
<linear_to_mel_weight_matrix/linspace_1/strided_slice_2/stackPack0linear_to_mel_weight_matrix/linspace_1/add_2:z:0*
N*
T0*
_output_shapes
:�
>linear_to_mel_weight_matrix/linspace_1/strided_slice_2/stack_1Pack7linear_to_mel_weight_matrix/linspace_1/Const_1:output:0*
N*
T0*
_output_shapes
:�
>linear_to_mel_weight_matrix/linspace_1/strided_slice_2/stack_2Pack7linear_to_mel_weight_matrix/linspace_1/Const_2:output:0*
N*
T0*
_output_shapes
:�
6linear_to_mel_weight_matrix/linspace_1/strided_slice_2StridedSlice7linear_to_mel_weight_matrix/linspace_1/Shape_2:output:0Elinear_to_mel_weight_matrix/linspace_1/strided_slice_2/stack:output:0Glinear_to_mel_weight_matrix/linspace_1/strided_slice_2/stack_1:output:0Glinear_to_mel_weight_matrix/linspace_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_maskv
4linear_to_mel_weight_matrix/linspace_1/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
/linear_to_mel_weight_matrix/linspace_1/concat_1ConcatV2?linear_to_mel_weight_matrix/linspace_1/strided_slice_1:output:09linear_to_mel_weight_matrix/linspace_1/Reshape_1:output:0?linear_to_mel_weight_matrix/linspace_1/strided_slice_2:output:0=linear_to_mel_weight_matrix/linspace_1/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
,linear_to_mel_weight_matrix/linspace_1/SliceSlice6linear_to_mel_weight_matrix/linspace_1/concat:output:0:linear_to_mel_weight_matrix/linspace_1/zeros_like:output:08linear_to_mel_weight_matrix/linspace_1/concat_1:output:0*
Index0*
T0*
_output_shapes
:4p
.linear_to_mel_weight_matrix/frame/frame_lengthConst*
_output_shapes
: *
dtype0*
value	B :n
,linear_to_mel_weight_matrix/frame/frame_stepConst*
_output_shapes
: *
dtype0*
value	B :q
&linear_to_mel_weight_matrix/frame/axisConst*
_output_shapes
: *
dtype0*
valueB :
���������q
'linear_to_mel_weight_matrix/frame/ShapeConst*
_output_shapes
:*
dtype0*
valueB:4o
,linear_to_mel_weight_matrix/frame/Size/ConstConst*
_output_shapes
: *
dtype0*
valueB h
&linear_to_mel_weight_matrix/frame/SizeConst*
_output_shapes
: *
dtype0*
value	B : q
.linear_to_mel_weight_matrix/frame/Size_1/ConstConst*
_output_shapes
: *
dtype0*
valueB j
(linear_to_mel_weight_matrix/frame/Size_1Const*
_output_shapes
: *
dtype0*
value	B : i
'linear_to_mel_weight_matrix/frame/ConstConst*
_output_shapes
: *
dtype0*
value	B : i
'linear_to_mel_weight_matrix/frame/sub/xConst*
_output_shapes
: *
dtype0*
value	B :4�
%linear_to_mel_weight_matrix/frame/subSub0linear_to_mel_weight_matrix/frame/sub/x:output:07linear_to_mel_weight_matrix/frame/frame_length:output:0*
T0*
_output_shapes
: �
*linear_to_mel_weight_matrix/frame/floordivFloorDiv)linear_to_mel_weight_matrix/frame/sub:z:05linear_to_mel_weight_matrix/frame/frame_step:output:0*
T0*
_output_shapes
: i
'linear_to_mel_weight_matrix/frame/add/xConst*
_output_shapes
: *
dtype0*
value	B :�
%linear_to_mel_weight_matrix/frame/addAddV20linear_to_mel_weight_matrix/frame/add/x:output:0.linear_to_mel_weight_matrix/frame/floordiv:z:0*
T0*
_output_shapes
: �
)linear_to_mel_weight_matrix/frame/MaximumMaximum0linear_to_mel_weight_matrix/frame/Const:output:0)linear_to_mel_weight_matrix/frame/add:z:0*
T0*
_output_shapes
: m
+linear_to_mel_weight_matrix/frame/gcd/ConstConst*
_output_shapes
: *
dtype0*
value	B :p
.linear_to_mel_weight_matrix/frame/floordiv_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
,linear_to_mel_weight_matrix/frame/floordiv_1FloorDiv7linear_to_mel_weight_matrix/frame/frame_length:output:07linear_to_mel_weight_matrix/frame/floordiv_1/y:output:0*
T0*
_output_shapes
: p
.linear_to_mel_weight_matrix/frame/floordiv_2/yConst*
_output_shapes
: *
dtype0*
value	B :�
,linear_to_mel_weight_matrix/frame/floordiv_2FloorDiv5linear_to_mel_weight_matrix/frame/frame_step:output:07linear_to_mel_weight_matrix/frame/floordiv_2/y:output:0*
T0*
_output_shapes
: t
1linear_to_mel_weight_matrix/frame/concat/values_0Const*
_output_shapes
: *
dtype0*
valueB {
1linear_to_mel_weight_matrix/frame/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:4t
1linear_to_mel_weight_matrix/frame/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB o
-linear_to_mel_weight_matrix/frame/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
(linear_to_mel_weight_matrix/frame/concatConcatV2:linear_to_mel_weight_matrix/frame/concat/values_0:output:0:linear_to_mel_weight_matrix/frame/concat/values_1:output:0:linear_to_mel_weight_matrix/frame/concat/values_2:output:06linear_to_mel_weight_matrix/frame/concat/axis:output:0*
N*
T0*
_output_shapes
:v
3linear_to_mel_weight_matrix/frame/concat_1/values_0Const*
_output_shapes
: *
dtype0*
valueB �
3linear_to_mel_weight_matrix/frame/concat_1/values_1Const*
_output_shapes
:*
dtype0*
valueB"4      v
3linear_to_mel_weight_matrix/frame/concat_1/values_2Const*
_output_shapes
: *
dtype0*
valueB q
/linear_to_mel_weight_matrix/frame/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
*linear_to_mel_weight_matrix/frame/concat_1ConcatV2<linear_to_mel_weight_matrix/frame/concat_1/values_0:output:0<linear_to_mel_weight_matrix/frame/concat_1/values_1:output:0<linear_to_mel_weight_matrix/frame/concat_1/values_2:output:08linear_to_mel_weight_matrix/frame/concat_1/axis:output:0*
N*
T0*
_output_shapes
:}
3linear_to_mel_weight_matrix/frame/zeros_like/tensorConst*
_output_shapes
:*
dtype0*
valueB:4v
,linear_to_mel_weight_matrix/frame/zeros_likeConst*
_output_shapes
:*
dtype0*
valueB: �
Alinear_to_mel_weight_matrix/frame/ones_like/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:s
1linear_to_mel_weight_matrix/frame/ones_like/ConstConst*
_output_shapes
: *
dtype0*
value	B :�
+linear_to_mel_weight_matrix/frame/ones_likeFillJlinear_to_mel_weight_matrix/frame/ones_like/Shape/shape_as_tensor:output:0:linear_to_mel_weight_matrix/frame/ones_like/Const:output:0*
T0*
_output_shapes
:�
.linear_to_mel_weight_matrix/frame/StridedSliceStridedSlice5linear_to_mel_weight_matrix/linspace_1/Slice:output:05linear_to_mel_weight_matrix/frame/zeros_like:output:01linear_to_mel_weight_matrix/frame/concat:output:04linear_to_mel_weight_matrix/frame/ones_like:output:0*
Index0*
T0*
_output_shapes
:4�
)linear_to_mel_weight_matrix/frame/ReshapeReshape7linear_to_mel_weight_matrix/frame/StridedSlice:output:03linear_to_mel_weight_matrix/frame/concat_1:output:0*
T0*
_output_shapes

:4o
-linear_to_mel_weight_matrix/frame/range/startConst*
_output_shapes
: *
dtype0*
value	B : o
-linear_to_mel_weight_matrix/frame/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
'linear_to_mel_weight_matrix/frame/rangeRange6linear_to_mel_weight_matrix/frame/range/start:output:0-linear_to_mel_weight_matrix/frame/Maximum:z:06linear_to_mel_weight_matrix/frame/range/delta:output:0*
_output_shapes
:2�
%linear_to_mel_weight_matrix/frame/mulMul0linear_to_mel_weight_matrix/frame/range:output:00linear_to_mel_weight_matrix/frame/floordiv_2:z:0*
T0*
_output_shapes
:2u
3linear_to_mel_weight_matrix/frame/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :�
1linear_to_mel_weight_matrix/frame/Reshape_1/shapePack-linear_to_mel_weight_matrix/frame/Maximum:z:0<linear_to_mel_weight_matrix/frame/Reshape_1/shape/1:output:0*
N*
T0*
_output_shapes
:�
+linear_to_mel_weight_matrix/frame/Reshape_1Reshape)linear_to_mel_weight_matrix/frame/mul:z:0:linear_to_mel_weight_matrix/frame/Reshape_1/shape:output:0*
T0*
_output_shapes

:2q
/linear_to_mel_weight_matrix/frame/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : q
/linear_to_mel_weight_matrix/frame/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
)linear_to_mel_weight_matrix/frame/range_1Range8linear_to_mel_weight_matrix/frame/range_1/start:output:00linear_to_mel_weight_matrix/frame/floordiv_1:z:08linear_to_mel_weight_matrix/frame/range_1/delta:output:0*
_output_shapes
:u
3linear_to_mel_weight_matrix/frame/Reshape_2/shape/0Const*
_output_shapes
: *
dtype0*
value	B :�
1linear_to_mel_weight_matrix/frame/Reshape_2/shapePack<linear_to_mel_weight_matrix/frame/Reshape_2/shape/0:output:00linear_to_mel_weight_matrix/frame/floordiv_1:z:0*
N*
T0*
_output_shapes
:�
+linear_to_mel_weight_matrix/frame/Reshape_2Reshape2linear_to_mel_weight_matrix/frame/range_1:output:0:linear_to_mel_weight_matrix/frame/Reshape_2/shape:output:0*
T0*
_output_shapes

:�
'linear_to_mel_weight_matrix/frame/add_1AddV24linear_to_mel_weight_matrix/frame/Reshape_1:output:04linear_to_mel_weight_matrix/frame/Reshape_2:output:0*
T0*
_output_shapes

:2l
)linear_to_mel_weight_matrix/frame/Const_1Const*
_output_shapes
: *
dtype0*
valueB l
)linear_to_mel_weight_matrix/frame/Const_2Const*
_output_shapes
: *
dtype0*
valueB �
(linear_to_mel_weight_matrix/frame/packedPack-linear_to_mel_weight_matrix/frame/Maximum:z:07linear_to_mel_weight_matrix/frame/frame_length:output:0*
N*
T0*
_output_shapes
:q
/linear_to_mel_weight_matrix/frame/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
*linear_to_mel_weight_matrix/frame/GatherV2GatherV22linear_to_mel_weight_matrix/frame/Reshape:output:0+linear_to_mel_weight_matrix/frame/add_1:z:08linear_to_mel_weight_matrix/frame/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*"
_output_shapes
:2q
/linear_to_mel_weight_matrix/frame/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
*linear_to_mel_weight_matrix/frame/concat_2ConcatV22linear_to_mel_weight_matrix/frame/Const_1:output:01linear_to_mel_weight_matrix/frame/packed:output:02linear_to_mel_weight_matrix/frame/Const_2:output:08linear_to_mel_weight_matrix/frame/concat_2/axis:output:0*
N*
T0*
_output_shapes
:�
+linear_to_mel_weight_matrix/frame/Reshape_3Reshape3linear_to_mel_weight_matrix/frame/GatherV2:output:03linear_to_mel_weight_matrix/frame/concat_2:output:0*
T0*
_output_shapes

:2m
+linear_to_mel_weight_matrix/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
!linear_to_mel_weight_matrix/splitSplit4linear_to_mel_weight_matrix/split/split_dim:output:04linear_to_mel_weight_matrix/frame/Reshape_3:output:0*
T0*2
_output_shapes 
:2:2:2*
	num_splitz
)linear_to_mel_weight_matrix/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   2   �
#linear_to_mel_weight_matrix/ReshapeReshape*linear_to_mel_weight_matrix/split:output:02linear_to_mel_weight_matrix/Reshape/shape:output:0*
T0*
_output_shapes

:2|
+linear_to_mel_weight_matrix/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   2   �
%linear_to_mel_weight_matrix/Reshape_1Reshape*linear_to_mel_weight_matrix/split:output:14linear_to_mel_weight_matrix/Reshape_1/shape:output:0*
T0*
_output_shapes

:2|
+linear_to_mel_weight_matrix/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"   2   �
%linear_to_mel_weight_matrix/Reshape_2Reshape*linear_to_mel_weight_matrix/split:output:24linear_to_mel_weight_matrix/Reshape_2/shape:output:0*
T0*
_output_shapes

:2�
linear_to_mel_weight_matrix/subSub/linear_to_mel_weight_matrix/ExpandDims:output:0,linear_to_mel_weight_matrix/Reshape:output:0*
T0*
_output_shapes
:	�2�
!linear_to_mel_weight_matrix/sub_1Sub.linear_to_mel_weight_matrix/Reshape_1:output:0,linear_to_mel_weight_matrix/Reshape:output:0*
T0*
_output_shapes

:2�
%linear_to_mel_weight_matrix/truediv_1RealDiv#linear_to_mel_weight_matrix/sub:z:0%linear_to_mel_weight_matrix/sub_1:z:0*
T0*
_output_shapes
:	�2�
!linear_to_mel_weight_matrix/sub_2Sub.linear_to_mel_weight_matrix/Reshape_2:output:0/linear_to_mel_weight_matrix/ExpandDims:output:0*
T0*
_output_shapes
:	�2�
!linear_to_mel_weight_matrix/sub_3Sub.linear_to_mel_weight_matrix/Reshape_2:output:0.linear_to_mel_weight_matrix/Reshape_1:output:0*
T0*
_output_shapes

:2�
%linear_to_mel_weight_matrix/truediv_2RealDiv%linear_to_mel_weight_matrix/sub_2:z:0%linear_to_mel_weight_matrix/sub_3:z:0*
T0*
_output_shapes
:	�2�
#linear_to_mel_weight_matrix/MinimumMinimum)linear_to_mel_weight_matrix/truediv_1:z:0)linear_to_mel_weight_matrix/truediv_2:z:0*
T0*
_output_shapes
:	�2�
#linear_to_mel_weight_matrix/MaximumMaximum*linear_to_mel_weight_matrix/Const:output:0'linear_to_mel_weight_matrix/Minimum:z:0*
T0*
_output_shapes
:	�2�
$linear_to_mel_weight_matrix/paddingsConst*
_output_shapes

:*
dtype0*)
value B"               �
linear_to_mel_weight_matrixPad'linear_to_mel_weight_matrix/Maximum:z:0-linear_to_mel_weight_matrix/paddings:output:0*
T0*
_output_shapes
:	�2h
MatMulMatMulAbs:y:0$linear_to_mel_weight_matrix:output:0*
T0*
_output_shapes

:b2J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5W
addAddV2MatMul:product:0add/y:output:0*
T0*
_output_shapes

:b2<
LogLogadd:z:0*
T0*
_output_shapes

:b2Y
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������g

ExpandDims
ExpandDimsLog:y:0ExpandDims/dim:output:0*
T0*"
_output_shapes
:b2R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : {
ExpandDims_1
ExpandDimsExpandDims:output:0ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:b2�
StatefulPartitionedCallStatefulPartitionedCallExpandDims_1:output:0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37*3
Tin,
*2(*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*G
_read_only_resource_inputs)
'%	
 !"#$%&'*-
config_proto

CPU

GPU 2J 8� *3
f.R,
*__inference_signature_wrapper___call___753[
ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
���������{
ArgMaxArgMax StatefulPartitionedCall:output:0ArgMax/dimension:output:0*
T0*#
_output_shapes
:����������
GatherV2/paramsConst*
_output_shapes
:*
dtype0*[
valueRBPB
backgroundBdownBgoBleftBnoBoffBonBrightBstopBupByesBunknownO
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
GatherV2GatherV2GatherV2/params:output:0ArgMax:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:���������Z
IdentityIdentityArgMax:output:0^NoOp*
T0	*#
_output_shapes
:���������^

Identity_1IdentityGatherV2:output:0^NoOp*
T0*#
_output_shapes
:���������q

Identity_2Identity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesr
p: :b2:b2: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:$' 

_user_specified_name1902:$& 

_user_specified_name1900:$% 

_user_specified_name1898:$$ 

_user_specified_name1896:$# 

_user_specified_name1894:$" 

_user_specified_name1892:$! 

_user_specified_name1890:$  

_user_specified_name1888:$ 

_user_specified_name1886:$ 

_user_specified_name1884:$ 

_user_specified_name1882:$ 

_user_specified_name1880:$ 

_user_specified_name1878:$ 

_user_specified_name1876:$ 

_user_specified_name1874:$ 

_user_specified_name1872:$ 

_user_specified_name1870:$ 

_user_specified_name1868:$ 

_user_specified_name1866:$ 

_user_specified_name1864:$ 

_user_specified_name1862:$ 

_user_specified_name1860:$ 

_user_specified_name1858:$ 

_user_specified_name1856:$ 

_user_specified_name1854:$ 

_user_specified_name1852:$ 

_user_specified_name1850:$ 

_user_specified_name1848:$ 

_user_specified_name1846:$
 

_user_specified_name1844:$	 

_user_specified_name1842:$ 

_user_specified_name1840:$ 

_user_specified_name1838:$ 

_user_specified_name1836:$ 

_user_specified_name1834:$ 

_user_specified_name1832:$ 

_user_specified_name1830:LH
&
_output_shapes
:b2

_user_specified_name1828:LH
&
_output_shapes
:b2

_user_specified_name1826:= 9

_output_shapes
: 

_user_specified_nameinput
��
�1
 __inference__traced_restore_4281
file_prefix7
!assignvariableop_imageinput__mean:b2=
'assignvariableop_1_imageinput__variance:b2.
$assignvariableop_2_imageinput__count:	 =
#assignvariableop_3_conv_1__kernel_1:/
!assignvariableop_4_conv_1__bias_1:5
'assignvariableop_5_batchnorm_1__gamma_1:4
&assignvariableop_6_batchnorm_1__beta_1:;
-assignvariableop_7_batchnorm_1__moving_mean_1:?
1assignvariableop_8_batchnorm_1__moving_variance_1:=
#assignvariableop_9_conv_2__kernel_1:0
"assignvariableop_10_conv_2__bias_1:6
(assignvariableop_11_batchnorm_2__gamma_1:5
'assignvariableop_12_batchnorm_2__beta_1:<
.assignvariableop_13_batchnorm_2__moving_mean_1:@
2assignvariableop_14_batchnorm_2__moving_variance_1:>
$assignvariableop_15_conv_3__kernel_1:0
"assignvariableop_16_conv_3__bias_1:6
(assignvariableop_17_batchnorm_3__gamma_1:5
'assignvariableop_18_batchnorm_3__beta_1:<
.assignvariableop_19_batchnorm_3__moving_mean_1:@
2assignvariableop_20_batchnorm_3__moving_variance_1:>
$assignvariableop_21_conv_4__kernel_1: 0
"assignvariableop_22_conv_4__bias_1: 6
(assignvariableop_23_batchnorm_4__gamma_1: 5
'assignvariableop_24_batchnorm_4__beta_1: <
.assignvariableop_25_batchnorm_4__moving_mean_1: @
2assignvariableop_26_batchnorm_4__moving_variance_1: >
$assignvariableop_27_conv_5__kernel_1: P0
"assignvariableop_28_conv_5__bias_1:P6
(assignvariableop_29_batchnorm_5__gamma_1:P5
'assignvariableop_30_batchnorm_5__beta_1:P<
.assignvariableop_31_batchnorm_5__moving_mean_1:P@
2assignvariableop_32_batchnorm_5__moving_variance_1:P@
,assignvariableop_33_lstm__lstm_cell_kernel_1:
��J
6assignvariableop_34_lstm__lstm_cell_recurrent_kernel_1:
��9
*assignvariableop_35_lstm__lstm_cell_bias_1:	�8
*assignvariableop_36_seed_generator_state_1:5
"assignvariableop_37_fc_1__kernel_1:	�.
 assignvariableop_38_fc_1__bias_1:6
(assignvariableop_39_seed_generator_state:4
"assignvariableop_40_fc_2__kernel_1:.
 assignvariableop_41_fc_2__bias_1:3
%assignvariableop_42_batchnorm_1__beta:<
"assignvariableop_43_conv_2__kernel:3
%assignvariableop_44_batchnorm_3__beta:<
"assignvariableop_45_conv_4__kernel: .
 assignvariableop_46_conv_5__bias:P<
"assignvariableop_47_conv_1__kernel:.
 assignvariableop_48_conv_3__bias:<
"assignvariableop_49_conv_5__kernel: P<
"assignvariableop_50_conv_3__kernel:>
*assignvariableop_51_lstm__lstm_cell_kernel:
��3
 assignvariableop_52_fc_1__kernel:	�.
 assignvariableop_53_conv_1__bias:4
&assignvariableop_54_batchnorm_2__gamma:4
&assignvariableop_55_batchnorm_4__gamma: 4
&assignvariableop_56_batchnorm_5__gamma:PH
4assignvariableop_57_lstm__lstm_cell_recurrent_kernel:
��2
 assignvariableop_58_fc_2__kernel:7
(assignvariableop_59_lstm__lstm_cell_bias:	�,
assignvariableop_60_fc_1__bias:4
&assignvariableop_61_batchnorm_3__gamma:,
assignvariableop_62_fc_2__bias:3
%assignvariableop_63_batchnorm_2__beta:3
%assignvariableop_64_batchnorm_4__beta: 3
%assignvariableop_65_batchnorm_5__beta:P4
&assignvariableop_66_batchnorm_1__gamma:.
 assignvariableop_67_conv_2__bias:.
 assignvariableop_68_conv_4__bias: >
0assignvariableop_69_batchnorm_5__moving_variance:P>
0assignvariableop_70_batchnorm_2__moving_variance:>
0assignvariableop_71_batchnorm_1__moving_variance::
,assignvariableop_72_batchnorm_2__moving_mean::
,assignvariableop_73_batchnorm_4__moving_mean: :
,assignvariableop_74_batchnorm_5__moving_mean:P>
0assignvariableop_75_batchnorm_4__moving_variance: :
,assignvariableop_76_batchnorm_1__moving_mean::
,assignvariableop_77_batchnorm_3__moving_mean:>
0assignvariableop_78_batchnorm_3__moving_variance:
identity_80��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_61�AssignVariableOp_62�AssignVariableOp_63�AssignVariableOp_64�AssignVariableOp_65�AssignVariableOp_66�AssignVariableOp_67�AssignVariableOp_68�AssignVariableOp_69�AssignVariableOp_7�AssignVariableOp_70�AssignVariableOp_71�AssignVariableOp_72�AssignVariableOp_73�AssignVariableOp_74�AssignVariableOp_75�AssignVariableOp_76�AssignVariableOp_77�AssignVariableOp_78�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:P*
dtype0*�
value�B�PB,model/variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model/variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model/variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model/variables/3/.ATTRIBUTES/VARIABLE_VALUEB,model/variables/4/.ATTRIBUTES/VARIABLE_VALUEB,model/variables/5/.ATTRIBUTES/VARIABLE_VALUEB,model/variables/6/.ATTRIBUTES/VARIABLE_VALUEB,model/variables/7/.ATTRIBUTES/VARIABLE_VALUEB,model/variables/8/.ATTRIBUTES/VARIABLE_VALUEB,model/variables/9/.ATTRIBUTES/VARIABLE_VALUEB-model/variables/10/.ATTRIBUTES/VARIABLE_VALUEB-model/variables/11/.ATTRIBUTES/VARIABLE_VALUEB-model/variables/12/.ATTRIBUTES/VARIABLE_VALUEB-model/variables/13/.ATTRIBUTES/VARIABLE_VALUEB-model/variables/14/.ATTRIBUTES/VARIABLE_VALUEB-model/variables/15/.ATTRIBUTES/VARIABLE_VALUEB-model/variables/16/.ATTRIBUTES/VARIABLE_VALUEB-model/variables/17/.ATTRIBUTES/VARIABLE_VALUEB-model/variables/18/.ATTRIBUTES/VARIABLE_VALUEB-model/variables/19/.ATTRIBUTES/VARIABLE_VALUEB-model/variables/20/.ATTRIBUTES/VARIABLE_VALUEB-model/variables/21/.ATTRIBUTES/VARIABLE_VALUEB-model/variables/22/.ATTRIBUTES/VARIABLE_VALUEB-model/variables/23/.ATTRIBUTES/VARIABLE_VALUEB-model/variables/24/.ATTRIBUTES/VARIABLE_VALUEB-model/variables/25/.ATTRIBUTES/VARIABLE_VALUEB-model/variables/26/.ATTRIBUTES/VARIABLE_VALUEB-model/variables/27/.ATTRIBUTES/VARIABLE_VALUEB-model/variables/28/.ATTRIBUTES/VARIABLE_VALUEB-model/variables/29/.ATTRIBUTES/VARIABLE_VALUEB-model/variables/30/.ATTRIBUTES/VARIABLE_VALUEB-model/variables/31/.ATTRIBUTES/VARIABLE_VALUEB-model/variables/32/.ATTRIBUTES/VARIABLE_VALUEB-model/variables/33/.ATTRIBUTES/VARIABLE_VALUEB-model/variables/34/.ATTRIBUTES/VARIABLE_VALUEB-model/variables/35/.ATTRIBUTES/VARIABLE_VALUEB-model/variables/36/.ATTRIBUTES/VARIABLE_VALUEB-model/variables/37/.ATTRIBUTES/VARIABLE_VALUEB-model/variables/38/.ATTRIBUTES/VARIABLE_VALUEB-model/variables/39/.ATTRIBUTES/VARIABLE_VALUEB-model/variables/40/.ATTRIBUTES/VARIABLE_VALUEB-model/variables/41/.ATTRIBUTES/VARIABLE_VALUEB1model/_all_variables/0/.ATTRIBUTES/VARIABLE_VALUEB1model/_all_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1model/_all_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1model/_all_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1model/_all_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1model/_all_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1model/_all_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1model/_all_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1model/_all_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1model/_all_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2model/_all_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2model/_all_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2model/_all_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2model/_all_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2model/_all_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2model/_all_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2model/_all_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2model/_all_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2model/_all_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2model/_all_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2model/_all_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2model/_all_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2model/_all_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2model/_all_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2model/_all_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2model/_all_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2model/_all_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2model/_all_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2model/_all_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2model/_all_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2model/_all_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2model/_all_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2model/_all_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2model/_all_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2model/_all_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2model/_all_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2model/_all_variables/36/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:P*
dtype0*�
value�B�PB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*^
dtypesT
R2P	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp!assignvariableop_imageinput__meanIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp'assignvariableop_1_imageinput__varianceIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp$assignvariableop_2_imageinput__countIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp#assignvariableop_3_conv_1__kernel_1Identity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp!assignvariableop_4_conv_1__bias_1Identity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp'assignvariableop_5_batchnorm_1__gamma_1Identity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp&assignvariableop_6_batchnorm_1__beta_1Identity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp-assignvariableop_7_batchnorm_1__moving_mean_1Identity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp1assignvariableop_8_batchnorm_1__moving_variance_1Identity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp#assignvariableop_9_conv_2__kernel_1Identity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp"assignvariableop_10_conv_2__bias_1Identity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp(assignvariableop_11_batchnorm_2__gamma_1Identity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp'assignvariableop_12_batchnorm_2__beta_1Identity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp.assignvariableop_13_batchnorm_2__moving_mean_1Identity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp2assignvariableop_14_batchnorm_2__moving_variance_1Identity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp$assignvariableop_15_conv_3__kernel_1Identity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp"assignvariableop_16_conv_3__bias_1Identity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp(assignvariableop_17_batchnorm_3__gamma_1Identity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp'assignvariableop_18_batchnorm_3__beta_1Identity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp.assignvariableop_19_batchnorm_3__moving_mean_1Identity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp2assignvariableop_20_batchnorm_3__moving_variance_1Identity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp$assignvariableop_21_conv_4__kernel_1Identity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp"assignvariableop_22_conv_4__bias_1Identity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp(assignvariableop_23_batchnorm_4__gamma_1Identity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp'assignvariableop_24_batchnorm_4__beta_1Identity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp.assignvariableop_25_batchnorm_4__moving_mean_1Identity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp2assignvariableop_26_batchnorm_4__moving_variance_1Identity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp$assignvariableop_27_conv_5__kernel_1Identity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp"assignvariableop_28_conv_5__bias_1Identity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp(assignvariableop_29_batchnorm_5__gamma_1Identity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp'assignvariableop_30_batchnorm_5__beta_1Identity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp.assignvariableop_31_batchnorm_5__moving_mean_1Identity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp2assignvariableop_32_batchnorm_5__moving_variance_1Identity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp,assignvariableop_33_lstm__lstm_cell_kernel_1Identity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp6assignvariableop_34_lstm__lstm_cell_recurrent_kernel_1Identity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp*assignvariableop_35_lstm__lstm_cell_bias_1Identity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp*assignvariableop_36_seed_generator_state_1Identity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp"assignvariableop_37_fc_1__kernel_1Identity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp assignvariableop_38_fc_1__bias_1Identity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp(assignvariableop_39_seed_generator_stateIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp"assignvariableop_40_fc_2__kernel_1Identity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp assignvariableop_41_fc_2__bias_1Identity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp%assignvariableop_42_batchnorm_1__betaIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp"assignvariableop_43_conv_2__kernelIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp%assignvariableop_44_batchnorm_3__betaIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp"assignvariableop_45_conv_4__kernelIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp assignvariableop_46_conv_5__biasIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOp"assignvariableop_47_conv_1__kernelIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOp assignvariableop_48_conv_3__biasIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOp"assignvariableop_49_conv_5__kernelIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOp"assignvariableop_50_conv_3__kernelIdentity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOp*assignvariableop_51_lstm__lstm_cell_kernelIdentity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOp assignvariableop_52_fc_1__kernelIdentity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOp assignvariableop_53_conv_1__biasIdentity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOp&assignvariableop_54_batchnorm_2__gammaIdentity_54:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOp&assignvariableop_55_batchnorm_4__gammaIdentity_55:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_56AssignVariableOp&assignvariableop_56_batchnorm_5__gammaIdentity_56:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_57AssignVariableOp4assignvariableop_57_lstm__lstm_cell_recurrent_kernelIdentity_57:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_58AssignVariableOp assignvariableop_58_fc_2__kernelIdentity_58:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_59AssignVariableOp(assignvariableop_59_lstm__lstm_cell_biasIdentity_59:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_60AssignVariableOpassignvariableop_60_fc_1__biasIdentity_60:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_61AssignVariableOp&assignvariableop_61_batchnorm_3__gammaIdentity_61:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_62AssignVariableOpassignvariableop_62_fc_2__biasIdentity_62:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_63AssignVariableOp%assignvariableop_63_batchnorm_2__betaIdentity_63:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_64AssignVariableOp%assignvariableop_64_batchnorm_4__betaIdentity_64:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_65AssignVariableOp%assignvariableop_65_batchnorm_5__betaIdentity_65:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_66AssignVariableOp&assignvariableop_66_batchnorm_1__gammaIdentity_66:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_67AssignVariableOp assignvariableop_67_conv_2__biasIdentity_67:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_68AssignVariableOp assignvariableop_68_conv_4__biasIdentity_68:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_69AssignVariableOp0assignvariableop_69_batchnorm_5__moving_varianceIdentity_69:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_70AssignVariableOp0assignvariableop_70_batchnorm_2__moving_varianceIdentity_70:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_71AssignVariableOp0assignvariableop_71_batchnorm_1__moving_varianceIdentity_71:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_72AssignVariableOp,assignvariableop_72_batchnorm_2__moving_meanIdentity_72:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_73AssignVariableOp,assignvariableop_73_batchnorm_4__moving_meanIdentity_73:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_74AssignVariableOp,assignvariableop_74_batchnorm_5__moving_meanIdentity_74:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_75AssignVariableOp0assignvariableop_75_batchnorm_4__moving_varianceIdentity_75:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_76AssignVariableOp,assignvariableop_76_batchnorm_1__moving_meanIdentity_76:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_77AssignVariableOp,assignvariableop_77_batchnorm_3__moving_meanIdentity_77:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_78AssignVariableOp0assignvariableop_78_batchnorm_3__moving_varianceIdentity_78:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_79Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_80IdentityIdentity_79:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_80Identity_80:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:<O8
6
_user_specified_namebatchnorm_3_/moving_variance:8N4
2
_user_specified_namebatchnorm_3_/moving_mean:8M4
2
_user_specified_namebatchnorm_1_/moving_mean:<L8
6
_user_specified_namebatchnorm_4_/moving_variance:8K4
2
_user_specified_namebatchnorm_5_/moving_mean:8J4
2
_user_specified_namebatchnorm_4_/moving_mean:8I4
2
_user_specified_namebatchnorm_2_/moving_mean:<H8
6
_user_specified_namebatchnorm_1_/moving_variance:<G8
6
_user_specified_namebatchnorm_2_/moving_variance:<F8
6
_user_specified_namebatchnorm_5_/moving_variance:,E(
&
_user_specified_nameconv_4_/bias:,D(
&
_user_specified_nameconv_2_/bias:2C.
,
_user_specified_namebatchnorm_1_/gamma:1B-
+
_user_specified_namebatchnorm_5_/beta:1A-
+
_user_specified_namebatchnorm_4_/beta:1@-
+
_user_specified_namebatchnorm_2_/beta:*?&
$
_user_specified_name
fc_2_/bias:2>.
,
_user_specified_namebatchnorm_3_/gamma:*=&
$
_user_specified_name
fc_1_/bias:4<0
.
_user_specified_namelstm_/lstm_cell/bias:,;(
&
_user_specified_namefc_2_/kernel:@:<
:
_user_specified_name" lstm_/lstm_cell/recurrent_kernel:29.
,
_user_specified_namebatchnorm_5_/gamma:28.
,
_user_specified_namebatchnorm_4_/gamma:27.
,
_user_specified_namebatchnorm_2_/gamma:,6(
&
_user_specified_nameconv_1_/bias:,5(
&
_user_specified_namefc_1_/kernel:642
0
_user_specified_namelstm_/lstm_cell/kernel:.3*
(
_user_specified_nameconv_3_/kernel:.2*
(
_user_specified_nameconv_5_/kernel:,1(
&
_user_specified_nameconv_3_/bias:.0*
(
_user_specified_nameconv_1_/kernel:,/(
&
_user_specified_nameconv_5_/bias:..*
(
_user_specified_nameconv_4_/kernel:1--
+
_user_specified_namebatchnorm_3_/beta:.,*
(
_user_specified_nameconv_2_/kernel:1+-
+
_user_specified_namebatchnorm_1_/beta:,*(
&
_user_specified_namefc_2_/bias_1:.)*
(
_user_specified_namefc_2_/kernel_1:4(0
.
_user_specified_nameseed_generator_state:,'(
&
_user_specified_namefc_1_/bias_1:.&*
(
_user_specified_namefc_1_/kernel_1:6%2
0
_user_specified_nameseed_generator_state_1:6$2
0
_user_specified_namelstm_/lstm_cell/bias_1:B#>
<
_user_specified_name$"lstm_/lstm_cell/recurrent_kernel_1:8"4
2
_user_specified_namelstm_/lstm_cell/kernel_1:>!:
8
_user_specified_name batchnorm_5_/moving_variance_1:: 6
4
_user_specified_namebatchnorm_5_/moving_mean_1:3/
-
_user_specified_namebatchnorm_5_/beta_1:40
.
_user_specified_namebatchnorm_5_/gamma_1:.*
(
_user_specified_nameconv_5_/bias_1:0,
*
_user_specified_nameconv_5_/kernel_1:>:
8
_user_specified_name batchnorm_4_/moving_variance_1::6
4
_user_specified_namebatchnorm_4_/moving_mean_1:3/
-
_user_specified_namebatchnorm_4_/beta_1:40
.
_user_specified_namebatchnorm_4_/gamma_1:.*
(
_user_specified_nameconv_4_/bias_1:0,
*
_user_specified_nameconv_4_/kernel_1:>:
8
_user_specified_name batchnorm_3_/moving_variance_1::6
4
_user_specified_namebatchnorm_3_/moving_mean_1:3/
-
_user_specified_namebatchnorm_3_/beta_1:40
.
_user_specified_namebatchnorm_3_/gamma_1:.*
(
_user_specified_nameconv_3_/bias_1:0,
*
_user_specified_nameconv_3_/kernel_1:>:
8
_user_specified_name batchnorm_2_/moving_variance_1::6
4
_user_specified_namebatchnorm_2_/moving_mean_1:3/
-
_user_specified_namebatchnorm_2_/beta_1:40
.
_user_specified_namebatchnorm_2_/gamma_1:.*
(
_user_specified_nameconv_2_/bias_1:0
,
*
_user_specified_nameconv_2_/kernel_1:>	:
8
_user_specified_name batchnorm_1_/moving_variance_1::6
4
_user_specified_namebatchnorm_1_/moving_mean_1:3/
-
_user_specified_namebatchnorm_1_/beta_1:40
.
_user_specified_namebatchnorm_1_/gamma_1:.*
(
_user_specified_nameconv_1_/bias_1:0,
*
_user_specified_nameconv_1_/kernel_1:1-
+
_user_specified_nameimageinput_/count:40
.
_user_specified_nameimageinput_/variance:0,
*
_user_specified_nameimageinput_/mean:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
��
�	
__inference___call___3531	
input
unknown
	unknown_0#
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:$

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:$

unknown_19: 

unknown_20: 

unknown_21: 

unknown_22: 

unknown_23: 

unknown_24: $

unknown_25: P

unknown_26:P

unknown_27:P

unknown_28:P

unknown_29:P

unknown_30:P

unknown_31:
��

unknown_32:
��

unknown_33:	�

unknown_34:	�

unknown_35:

unknown_36:

unknown_37:
identity	

identity_1

identity_2��StatefulPartitionedCall<
SqueezeSqueezeinput*
T0*
_output_shapes
:H
ShapeShapeinput*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskH
sub/xConst*
_output_shapes
: *
dtype0*
value
B :�}S
subSubsub/x:output:0strided_slice:output:0*
T0*
_output_shapes
: K
	Maximum/xConst*
_output_shapes
: *
dtype0*
value	B : P
MaximumMaximumMaximum/x:output:0sub:z:0*
T0*
_output_shapes
: R
Pad/paddings/0/0Const*
_output_shapes
: *
dtype0*
value	B : l
Pad/paddings/0PackPad/paddings/0/0:output:0Maximum:z:0*
N*
T0*
_output_shapes
:_
Pad/paddingsPackPad/paddings/0:output:0*
N*
T0*
_output_shapes

:a
PadPadSqueeze:output:0Pad/paddings:output:0*
T0*#
_output_shapes
:���������Y
l2_normalize/SquareSquarePad:output:0*
T0*#
_output_shapes
:���������\
l2_normalize/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
l2_normalize/SumSuml2_normalize/Square:y:0l2_normalize/Const:output:0*
T0*
_output_shapes
:*
	keep_dims([
l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
l2_normalize/MaximumMaximuml2_normalize/Sum:output:0l2_normalize/Maximum/y:output:0*
T0*
_output_shapes
:Z
l2_normalize/RsqrtRsqrtl2_normalize/Maximum:z:0*
T0*
_output_shapes
:g
l2_normalizeMulPad:output:0l2_normalize/Rsqrt:y:0*
T0*#
_output_shapes
:���������T
stft/frame_lengthConst*
_output_shapes
: *
dtype0*
value
B :�R
stft/frame_stepConst*
_output_shapes
: *
dtype0*
value
B :�R
stft/fft_lengthConst*
_output_shapes
: *
dtype0*
value
B :�Z
stft/frame/axisConst*
_output_shapes
: *
dtype0*
valueB :
���������^
stft/frame/ShapeShapel2_normalize:z:0*
T0*
_output_shapes
::��Q
stft/frame/RankConst*
_output_shapes
: *
dtype0*
value	B :X
stft/frame/range/startConst*
_output_shapes
: *
dtype0*
value	B : X
stft/frame/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
stft/frame/rangeRangestft/frame/range/start:output:0stft/frame/Rank:output:0stft/frame/range/delta:output:0*
_output_shapes
:q
stft/frame/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������j
 stft/frame/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: j
 stft/frame/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
stft/frame/strided_sliceStridedSlicestft/frame/range:output:0'stft/frame/strided_slice/stack:output:0)stft/frame/strided_slice/stack_1:output:0)stft/frame/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskR
stft/frame/sub/yConst*
_output_shapes
: *
dtype0*
value	B :k
stft/frame/subSubstft/frame/Rank:output:0stft/frame/sub/y:output:0*
T0*
_output_shapes
: o
stft/frame/sub_1Substft/frame/sub:z:0!stft/frame/strided_slice:output:0*
T0*
_output_shapes
: U
stft/frame/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
stft/frame/packedPack!stft/frame/strided_slice:output:0stft/frame/packed/1:output:0stft/frame/sub_1:z:0*
N*
T0*
_output_shapes
:\
stft/frame/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : �
stft/frame/splitSplitVstft/frame/Shape:output:0stft/frame/packed:output:0#stft/frame/split/split_dim:output:0*

Tlen0*
T0*"
_output_shapes
: :: *
	num_split[
stft/frame/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB ]
stft/frame/Reshape/shape_1Const*
_output_shapes
: *
dtype0*
valueB ~
stft/frame/ReshapeReshapestft/frame/split:output:1#stft/frame/Reshape/shape_1:output:0*
T0*
_output_shapes
: Q
stft/frame/SizeConst*
_output_shapes
: *
dtype0*
value	B : S
stft/frame/Size_1Const*
_output_shapes
: *
dtype0*
value	B : R
stft/frame/ConstConst*
_output_shapes
: *
dtype0*
value	B : q
stft/frame/sub_2Substft/frame/Reshape:output:0stft/frame_length:output:0*
T0*
_output_shapes
: p
stft/frame/floordivFloorDivstft/frame/sub_2:z:0stft/frame_step:output:0*
T0*
_output_shapes
: R
stft/frame/add/xConst*
_output_shapes
: *
dtype0*
value	B :l
stft/frame/addAddV2stft/frame/add/x:output:0stft/frame/floordiv:z:0*
T0*
_output_shapes
: m
stft/frame/MaximumMaximumstft/frame/Const:output:0stft/frame/add:z:0*
T0*
_output_shapes
: V
stft/frame/gcd/ConstConst*
_output_shapes
: *
dtype0*
value	B :PY
stft/frame/floordiv_1/yConst*
_output_shapes
: *
dtype0*
value	B :P�
stft/frame/floordiv_1FloorDivstft/frame_length:output:0 stft/frame/floordiv_1/y:output:0*
T0*
_output_shapes
: Y
stft/frame/floordiv_2/yConst*
_output_shapes
: *
dtype0*
value	B :P~
stft/frame/floordiv_2FloorDivstft/frame_step:output:0 stft/frame/floordiv_2/y:output:0*
T0*
_output_shapes
: Y
stft/frame/floordiv_3/yConst*
_output_shapes
: *
dtype0*
value	B :P�
stft/frame/floordiv_3FloorDivstft/frame/Reshape:output:0 stft/frame/floordiv_3/y:output:0*
T0*
_output_shapes
: R
stft/frame/mul/yConst*
_output_shapes
: *
dtype0*
value	B :Pl
stft/frame/mulMulstft/frame/floordiv_3:z:0stft/frame/mul/y:output:0*
T0*
_output_shapes
: d
stft/frame/concat/values_1Packstft/frame/mul:z:0*
N*
T0*
_output_shapes
:X
stft/frame/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
stft/frame/concatConcatV2stft/frame/split:output:0#stft/frame/concat/values_1:output:0stft/frame/split:output:2stft/frame/concat/axis:output:0*
N*
T0*
_output_shapes
:`
stft/frame/concat_1/values_1/1Const*
_output_shapes
: *
dtype0*
value	B :P�
stft/frame/concat_1/values_1Packstft/frame/floordiv_3:z:0'stft/frame/concat_1/values_1/1:output:0*
N*
T0*
_output_shapes
:Z
stft/frame/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
stft/frame/concat_1ConcatV2stft/frame/split:output:0%stft/frame/concat_1/values_1:output:0stft/frame/split:output:2!stft/frame/concat_1/axis:output:0*
N*
T0*
_output_shapes
:_
stft/frame/zeros_likeConst*
_output_shapes
:*
dtype0*
valueB: t
*stft/frame/ones_like/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:\
stft/frame/ones_like/ConstConst*
_output_shapes
: *
dtype0*
value	B :�
stft/frame/ones_likeFill3stft/frame/ones_like/Shape/shape_as_tensor:output:0#stft/frame/ones_like/Const:output:0*
T0*
_output_shapes
:�
stft/frame/StridedSliceStridedSlicel2_normalize:z:0stft/frame/zeros_like:output:0stft/frame/concat:output:0stft/frame/ones_like:output:0*
Index0*
T0*#
_output_shapes
:����������
stft/frame/Reshape_1Reshape stft/frame/StridedSlice:output:0stft/frame/concat_1:output:0*
T0*'
_output_shapes
:���������PZ
stft/frame/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : Z
stft/frame/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
stft/frame/range_1Range!stft/frame/range_1/start:output:0stft/frame/Maximum:z:0!stft/frame/range_1/delta:output:0*#
_output_shapes
:���������}
stft/frame/mul_1Mulstft/frame/range_1:output:0stft/frame/floordiv_2:z:0*
T0*#
_output_shapes
:���������^
stft/frame/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :�
stft/frame/Reshape_2/shapePackstft/frame/Maximum:z:0%stft/frame/Reshape_2/shape/1:output:0*
N*
T0*
_output_shapes
:�
stft/frame/Reshape_2Reshapestft/frame/mul_1:z:0#stft/frame/Reshape_2/shape:output:0*
T0*'
_output_shapes
:���������Z
stft/frame/range_2/startConst*
_output_shapes
: *
dtype0*
value	B : Z
stft/frame/range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
stft/frame/range_2Range!stft/frame/range_2/start:output:0stft/frame/floordiv_1:z:0!stft/frame/range_2/delta:output:0*
_output_shapes
:^
stft/frame/Reshape_3/shape/0Const*
_output_shapes
: *
dtype0*
value	B :�
stft/frame/Reshape_3/shapePack%stft/frame/Reshape_3/shape/0:output:0stft/frame/floordiv_1:z:0*
N*
T0*
_output_shapes
:�
stft/frame/Reshape_3Reshapestft/frame/range_2:output:0#stft/frame/Reshape_3/shape:output:0*
T0*
_output_shapes

:�
stft/frame/add_1AddV2stft/frame/Reshape_2:output:0stft/frame/Reshape_3:output:0*
T0*'
_output_shapes
:���������}
stft/frame/packed_1Packstft/frame/Maximum:z:0stft/frame_length:output:0*
N*
T0*
_output_shapes
:�
stft/frame/GatherV2GatherV2stft/frame/Reshape_1:output:0stft/frame/add_1:z:0!stft/frame/strided_slice:output:0*
Taxis0*
Tindices0*
Tparams0*+
_output_shapes
:���������PZ
stft/frame/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
stft/frame/concat_2ConcatV2stft/frame/split:output:0stft/frame/packed_1:output:0stft/frame/split:output:2!stft/frame/concat_2/axis:output:0*
N*
T0*
_output_shapes
:�
stft/frame/Reshape_4Reshapestft/frame/GatherV2:output:0stft/frame/concat_2:output:0*
T0*(
_output_shapes
:����������[
stft/hann_window/periodicConst*
_output_shapes
: *
dtype0
*
value	B
 Zq
stft/hann_window/CastCast"stft/hann_window/periodic:output:0*

DstT0*

SrcT0
*
_output_shapes
: X
stft/hann_window/mod/yConst*
_output_shapes
: *
dtype0*
value	B :~
stft/hann_window/modFloorModstft/frame_length:output:0stft/hann_window/mod/y:output:0*
T0*
_output_shapes
: X
stft/hann_window/sub/xConst*
_output_shapes
: *
dtype0*
value	B :w
stft/hann_window/subSubstft/hann_window/sub/x:output:0stft/hann_window/mod:z:0*
T0*
_output_shapes
: q
stft/hann_window/mulMulstft/hann_window/Cast:y:0stft/hann_window/sub:z:0*
T0*
_output_shapes
: t
stft/hann_window/addAddV2stft/frame_length:output:0stft/hann_window/mul:z:0*
T0*
_output_shapes
: Z
stft/hann_window/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :{
stft/hann_window/sub_1Substft/hann_window/add:z:0!stft/hann_window/sub_1/y:output:0*
T0*
_output_shapes
: k
stft/hann_window/Cast_1Caststft/hann_window/sub_1:z:0*

DstT0*

SrcT0*
_output_shapes
: ^
stft/hann_window/range/startConst*
_output_shapes
: *
dtype0*
value	B : ^
stft/hann_window/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
stft/hann_window/rangeRange%stft/hann_window/range/start:output:0stft/frame_length:output:0%stft/hann_window/range/delta:output:0*
_output_shapes	
:�u
stft/hann_window/Cast_2Caststft/hann_window/range:output:0*

DstT0*

SrcT0*
_output_shapes	
:�[
stft/hann_window/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *��@�
stft/hann_window/mul_1Mulstft/hann_window/Const:output:0stft/hann_window/Cast_2:y:0*
T0*
_output_shapes	
:��
stft/hann_window/truedivRealDivstft/hann_window/mul_1:z:0stft/hann_window/Cast_1:y:0*
T0*
_output_shapes	
:�_
stft/hann_window/CosCosstft/hann_window/truediv:z:0*
T0*
_output_shapes	
:�]
stft/hann_window/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
stft/hann_window/mul_2Mul!stft/hann_window/mul_2/x:output:0stft/hann_window/Cos:y:0*
T0*
_output_shapes	
:�]
stft/hann_window/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
stft/hann_window/sub_2Sub!stft/hann_window/sub_2/x:output:0stft/hann_window/mul_2:z:0*
T0*
_output_shapes	
:�}
stft/mulMulstft/frame/Reshape_4:output:0stft/hann_window/sub_2:z:0*
T0*(
_output_shapes
:����������`
stft/rfft/packedPackstft/fft_length:output:0*
N*
T0*
_output_shapes
:w
stft/rfft/Pad/paddingsConst*
_output_shapes

:*
dtype0*)
value B"            p   v
stft/rfft/PadPadstft/mul:z:0stft/rfft/Pad/paddings:output:0*
T0*(
_output_shapes
:����������_
stft/rfft/fft_lengthConst*
_output_shapes
:*
dtype0*
valueB:�r
	stft/rfftRFFTstft/rfft/Pad:output:0stft/rfft/fft_length:output:0*(
_output_shapes
:����������O
Abs
ComplexAbsstft/rfft:output:0*(
_output_shapes
:����������L
Shape_1ShapeAbs:y:0*
T0*
_output_shapes
::��h
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
)linear_to_mel_weight_matrix/sample_rate/xConst*
_output_shapes
: *
dtype0*
value
B :�}�
'linear_to_mel_weight_matrix/sample_rateCast2linear_to_mel_weight_matrix/sample_rate/x:output:0*

DstT0*

SrcT0*
_output_shapes
: q
,linear_to_mel_weight_matrix/lower_edge_hertzConst*
_output_shapes
: *
dtype0*
valueB
 *  �Bq
,linear_to_mel_weight_matrix/upper_edge_hertzConst*
_output_shapes
: *
dtype0*
valueB
 * �mEf
!linear_to_mel_weight_matrix/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    j
%linear_to_mel_weight_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
#linear_to_mel_weight_matrix/truedivRealDiv+linear_to_mel_weight_matrix/sample_rate:y:0.linear_to_mel_weight_matrix/truediv/y:output:0*
T0*
_output_shapes
: {
)linear_to_mel_weight_matrix/linspace/CastCaststrided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: �
+linear_to_mel_weight_matrix/linspace/Cast_1Cast-linear_to_mel_weight_matrix/linspace/Cast:y:0*

DstT0*

SrcT0*
_output_shapes
: m
*linear_to_mel_weight_matrix/linspace/ShapeConst*
_output_shapes
: *
dtype0*
valueB o
,linear_to_mel_weight_matrix/linspace/Shape_1Const*
_output_shapes
: *
dtype0*
valueB �
2linear_to_mel_weight_matrix/linspace/BroadcastArgsBroadcastArgs3linear_to_mel_weight_matrix/linspace/Shape:output:05linear_to_mel_weight_matrix/linspace/Shape_1:output:0*
_output_shapes
: �
0linear_to_mel_weight_matrix/linspace/BroadcastToBroadcastTo*linear_to_mel_weight_matrix/Const:output:07linear_to_mel_weight_matrix/linspace/BroadcastArgs:r0:0*
T0*
_output_shapes
: �
2linear_to_mel_weight_matrix/linspace/BroadcastTo_1BroadcastTo'linear_to_mel_weight_matrix/truediv:z:07linear_to_mel_weight_matrix/linspace/BroadcastArgs:r0:0*
T0*
_output_shapes
: u
3linear_to_mel_weight_matrix/linspace/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : �
/linear_to_mel_weight_matrix/linspace/ExpandDims
ExpandDims9linear_to_mel_weight_matrix/linspace/BroadcastTo:output:0<linear_to_mel_weight_matrix/linspace/ExpandDims/dim:output:0*
T0*
_output_shapes
:w
5linear_to_mel_weight_matrix/linspace/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
1linear_to_mel_weight_matrix/linspace/ExpandDims_1
ExpandDims;linear_to_mel_weight_matrix/linspace/BroadcastTo_1:output:0>linear_to_mel_weight_matrix/linspace/ExpandDims_1/dim:output:0*
T0*
_output_shapes
:v
,linear_to_mel_weight_matrix/linspace/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:v
,linear_to_mel_weight_matrix/linspace/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:�
8linear_to_mel_weight_matrix/linspace/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
:linear_to_mel_weight_matrix/linspace/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
:linear_to_mel_weight_matrix/linspace/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
2linear_to_mel_weight_matrix/linspace/strided_sliceStridedSlice5linear_to_mel_weight_matrix/linspace/Shape_3:output:0Alinear_to_mel_weight_matrix/linspace/strided_slice/stack:output:0Clinear_to_mel_weight_matrix/linspace/strided_slice/stack_1:output:0Clinear_to_mel_weight_matrix/linspace/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
*linear_to_mel_weight_matrix/linspace/add/yConst*
_output_shapes
: *
dtype0*
value	B : �
(linear_to_mel_weight_matrix/linspace/addAddV2;linear_to_mel_weight_matrix/linspace/strided_slice:output:03linear_to_mel_weight_matrix/linspace/add/y:output:0*
T0*
_output_shapes
: y
7linear_to_mel_weight_matrix/linspace/SelectV2/conditionConst*
_output_shapes
: *
dtype0
*
value	B
 Zq
/linear_to_mel_weight_matrix/linspace/SelectV2/tConst*
_output_shapes
: *
dtype0*
value	B : �
-linear_to_mel_weight_matrix/linspace/SelectV2SelectV2@linear_to_mel_weight_matrix/linspace/SelectV2/condition:output:08linear_to_mel_weight_matrix/linspace/SelectV2/t:output:0,linear_to_mel_weight_matrix/linspace/add:z:0*
T0*
_output_shapes
: l
*linear_to_mel_weight_matrix/linspace/sub/yConst*
_output_shapes
: *
dtype0*
value	B :�
(linear_to_mel_weight_matrix/linspace/subSub-linear_to_mel_weight_matrix/linspace/Cast:y:03linear_to_mel_weight_matrix/linspace/sub/y:output:0*
T0*
_output_shapes
: p
.linear_to_mel_weight_matrix/linspace/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B : �
,linear_to_mel_weight_matrix/linspace/MaximumMaximum,linear_to_mel_weight_matrix/linspace/sub:z:07linear_to_mel_weight_matrix/linspace/Maximum/y:output:0*
T0*
_output_shapes
: n
,linear_to_mel_weight_matrix/linspace/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
*linear_to_mel_weight_matrix/linspace/sub_1Sub-linear_to_mel_weight_matrix/linspace/Cast:y:05linear_to_mel_weight_matrix/linspace/sub_1/y:output:0*
T0*
_output_shapes
: r
0linear_to_mel_weight_matrix/linspace/Maximum_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
.linear_to_mel_weight_matrix/linspace/Maximum_1Maximum.linear_to_mel_weight_matrix/linspace/sub_1:z:09linear_to_mel_weight_matrix/linspace/Maximum_1/y:output:0*
T0*
_output_shapes
: �
*linear_to_mel_weight_matrix/linspace/sub_2Sub:linear_to_mel_weight_matrix/linspace/ExpandDims_1:output:08linear_to_mel_weight_matrix/linspace/ExpandDims:output:0*
T0*
_output_shapes
:�
+linear_to_mel_weight_matrix/linspace/Cast_2Cast2linear_to_mel_weight_matrix/linspace/Maximum_1:z:0*

DstT0*

SrcT0*
_output_shapes
: �
,linear_to_mel_weight_matrix/linspace/truedivRealDiv.linear_to_mel_weight_matrix/linspace/sub_2:z:0/linear_to_mel_weight_matrix/linspace/Cast_2:y:0*
T0*
_output_shapes
:u
3linear_to_mel_weight_matrix/linspace/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : �
1linear_to_mel_weight_matrix/linspace/GreaterEqualGreaterEqual-linear_to_mel_weight_matrix/linspace/Cast:y:0<linear_to_mel_weight_matrix/linspace/GreaterEqual/y:output:0*
T0*
_output_shapes
: |
1linear_to_mel_weight_matrix/linspace/SelectV2_1/eConst*
_output_shapes
: *
dtype0*
valueB :
����������
/linear_to_mel_weight_matrix/linspace/SelectV2_1SelectV25linear_to_mel_weight_matrix/linspace/GreaterEqual:z:02linear_to_mel_weight_matrix/linspace/Maximum_1:z:0:linear_to_mel_weight_matrix/linspace/SelectV2_1/e:output:0*
T0*
_output_shapes
: r
0linear_to_mel_weight_matrix/linspace/range/startConst*
_output_shapes
: *
dtype0	*
value	B	 Rr
0linear_to_mel_weight_matrix/linspace/range/deltaConst*
_output_shapes
: *
dtype0	*
value	B	 R�
/linear_to_mel_weight_matrix/linspace/range/CastCast8linear_to_mel_weight_matrix/linspace/SelectV2_1:output:0*

DstT0	*

SrcT0*
_output_shapes
: �
*linear_to_mel_weight_matrix/linspace/rangeRange9linear_to_mel_weight_matrix/linspace/range/start:output:03linear_to_mel_weight_matrix/linspace/range/Cast:y:09linear_to_mel_weight_matrix/linspace/range/delta:output:0*

Tidx0	*
_output_shapes	
:��
+linear_to_mel_weight_matrix/linspace/Cast_3Cast3linear_to_mel_weight_matrix/linspace/range:output:0*

DstT0*

SrcT0	*
_output_shapes	
:�t
2linear_to_mel_weight_matrix/linspace/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : t
2linear_to_mel_weight_matrix/linspace/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
,linear_to_mel_weight_matrix/linspace/range_1Range;linear_to_mel_weight_matrix/linspace/range_1/start:output:0;linear_to_mel_weight_matrix/linspace/strided_slice:output:0;linear_to_mel_weight_matrix/linspace/range_1/delta:output:0*
_output_shapes
:�
*linear_to_mel_weight_matrix/linspace/EqualEqual6linear_to_mel_weight_matrix/linspace/SelectV2:output:05linear_to_mel_weight_matrix/linspace/range_1:output:0*
T0*
_output_shapes
:s
1linear_to_mel_weight_matrix/linspace/SelectV2_2/eConst*
_output_shapes
: *
dtype0*
value	B :�
/linear_to_mel_weight_matrix/linspace/SelectV2_2SelectV2.linear_to_mel_weight_matrix/linspace/Equal:z:00linear_to_mel_weight_matrix/linspace/Maximum:z:0:linear_to_mel_weight_matrix/linspace/SelectV2_2/e:output:0*
T0*
_output_shapes
:�
,linear_to_mel_weight_matrix/linspace/ReshapeReshape/linear_to_mel_weight_matrix/linspace/Cast_3:y:08linear_to_mel_weight_matrix/linspace/SelectV2_2:output:0*
T0*
_output_shapes	
:��
(linear_to_mel_weight_matrix/linspace/mulMul0linear_to_mel_weight_matrix/linspace/truediv:z:05linear_to_mel_weight_matrix/linspace/Reshape:output:0*
T0*
_output_shapes	
:��
*linear_to_mel_weight_matrix/linspace/add_1AddV28linear_to_mel_weight_matrix/linspace/ExpandDims:output:0,linear_to_mel_weight_matrix/linspace/mul:z:0*
T0*
_output_shapes	
:��
+linear_to_mel_weight_matrix/linspace/concatConcatV28linear_to_mel_weight_matrix/linspace/ExpandDims:output:0.linear_to_mel_weight_matrix/linspace/add_1:z:0:linear_to_mel_weight_matrix/linspace/ExpandDims_1:output:06linear_to_mel_weight_matrix/linspace/SelectV2:output:0*
N*
T0*
_output_shapes	
:�y
/linear_to_mel_weight_matrix/linspace/zeros_likeConst*
_output_shapes
:*
dtype0*
valueB: l
*linear_to_mel_weight_matrix/linspace/ConstConst*
_output_shapes
: *
dtype0*
value	B :�
:linear_to_mel_weight_matrix/linspace/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
<linear_to_mel_weight_matrix/linspace/strided_slice_1/stack_1Pack6linear_to_mel_weight_matrix/linspace/SelectV2:output:0*
N*
T0*
_output_shapes
:�
<linear_to_mel_weight_matrix/linspace/strided_slice_1/stack_2Pack3linear_to_mel_weight_matrix/linspace/Const:output:0*
N*
T0*
_output_shapes
:�
4linear_to_mel_weight_matrix/linspace/strided_slice_1StridedSlice5linear_to_mel_weight_matrix/linspace/Shape_2:output:0Clinear_to_mel_weight_matrix/linspace/strided_slice_1/stack:output:0Elinear_to_mel_weight_matrix/linspace/strided_slice_1/stack_1:output:0Elinear_to_mel_weight_matrix/linspace/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: ~
4linear_to_mel_weight_matrix/linspace/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:�
.linear_to_mel_weight_matrix/linspace/Reshape_1Reshape-linear_to_mel_weight_matrix/linspace/Cast:y:0=linear_to_mel_weight_matrix/linspace/Reshape_1/shape:output:0*
T0*
_output_shapes
:n
,linear_to_mel_weight_matrix/linspace/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :�
*linear_to_mel_weight_matrix/linspace/add_2AddV26linear_to_mel_weight_matrix/linspace/SelectV2:output:05linear_to_mel_weight_matrix/linspace/add_2/y:output:0*
T0*
_output_shapes
: n
,linear_to_mel_weight_matrix/linspace/Const_1Const*
_output_shapes
: *
dtype0*
value	B : n
,linear_to_mel_weight_matrix/linspace/Const_2Const*
_output_shapes
: *
dtype0*
value	B :�
:linear_to_mel_weight_matrix/linspace/strided_slice_2/stackPack.linear_to_mel_weight_matrix/linspace/add_2:z:0*
N*
T0*
_output_shapes
:�
<linear_to_mel_weight_matrix/linspace/strided_slice_2/stack_1Pack5linear_to_mel_weight_matrix/linspace/Const_1:output:0*
N*
T0*
_output_shapes
:�
<linear_to_mel_weight_matrix/linspace/strided_slice_2/stack_2Pack5linear_to_mel_weight_matrix/linspace/Const_2:output:0*
N*
T0*
_output_shapes
:�
4linear_to_mel_weight_matrix/linspace/strided_slice_2StridedSlice5linear_to_mel_weight_matrix/linspace/Shape_2:output:0Clinear_to_mel_weight_matrix/linspace/strided_slice_2/stack:output:0Elinear_to_mel_weight_matrix/linspace/strided_slice_2/stack_1:output:0Elinear_to_mel_weight_matrix/linspace/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_maskt
2linear_to_mel_weight_matrix/linspace/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
-linear_to_mel_weight_matrix/linspace/concat_1ConcatV2=linear_to_mel_weight_matrix/linspace/strided_slice_1:output:07linear_to_mel_weight_matrix/linspace/Reshape_1:output:0=linear_to_mel_weight_matrix/linspace/strided_slice_2:output:0;linear_to_mel_weight_matrix/linspace/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
*linear_to_mel_weight_matrix/linspace/SliceSlice4linear_to_mel_weight_matrix/linspace/concat:output:08linear_to_mel_weight_matrix/linspace/zeros_like:output:06linear_to_mel_weight_matrix/linspace/concat_1:output:0*
Index0*
T0*
_output_shapes	
:�y
/linear_to_mel_weight_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:{
1linear_to_mel_weight_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: {
1linear_to_mel_weight_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
)linear_to_mel_weight_matrix/strided_sliceStridedSlice3linear_to_mel_weight_matrix/linspace/Slice:output:08linear_to_mel_weight_matrix/strided_slice/stack:output:0:linear_to_mel_weight_matrix/strided_slice/stack_1:output:0:linear_to_mel_weight_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes	
:�*
end_maskw
2linear_to_mel_weight_matrix/hertz_to_mel/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  /D�
0linear_to_mel_weight_matrix/hertz_to_mel/truedivRealDiv2linear_to_mel_weight_matrix/strided_slice:output:0;linear_to_mel_weight_matrix/hertz_to_mel/truediv/y:output:0*
T0*
_output_shapes	
:�s
.linear_to_mel_weight_matrix/hertz_to_mel/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
,linear_to_mel_weight_matrix/hertz_to_mel/addAddV27linear_to_mel_weight_matrix/hertz_to_mel/add/x:output:04linear_to_mel_weight_matrix/hertz_to_mel/truediv:z:0*
T0*
_output_shapes	
:��
,linear_to_mel_weight_matrix/hertz_to_mel/LogLog0linear_to_mel_weight_matrix/hertz_to_mel/add:z:0*
T0*
_output_shapes	
:�s
.linear_to_mel_weight_matrix/hertz_to_mel/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 * ��D�
,linear_to_mel_weight_matrix/hertz_to_mel/mulMul7linear_to_mel_weight_matrix/hertz_to_mel/mul/x:output:00linear_to_mel_weight_matrix/hertz_to_mel/Log:y:0*
T0*
_output_shapes	
:�l
*linear_to_mel_weight_matrix/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
&linear_to_mel_weight_matrix/ExpandDims
ExpandDims0linear_to_mel_weight_matrix/hertz_to_mel/mul:z:03linear_to_mel_weight_matrix/ExpandDims/dim:output:0*
T0*
_output_shapes
:	�y
4linear_to_mel_weight_matrix/hertz_to_mel_1/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  /D�
2linear_to_mel_weight_matrix/hertz_to_mel_1/truedivRealDiv5linear_to_mel_weight_matrix/lower_edge_hertz:output:0=linear_to_mel_weight_matrix/hertz_to_mel_1/truediv/y:output:0*
T0*
_output_shapes
: u
0linear_to_mel_weight_matrix/hertz_to_mel_1/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
.linear_to_mel_weight_matrix/hertz_to_mel_1/addAddV29linear_to_mel_weight_matrix/hertz_to_mel_1/add/x:output:06linear_to_mel_weight_matrix/hertz_to_mel_1/truediv:z:0*
T0*
_output_shapes
: �
.linear_to_mel_weight_matrix/hertz_to_mel_1/LogLog2linear_to_mel_weight_matrix/hertz_to_mel_1/add:z:0*
T0*
_output_shapes
: u
0linear_to_mel_weight_matrix/hertz_to_mel_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 * ��D�
.linear_to_mel_weight_matrix/hertz_to_mel_1/mulMul9linear_to_mel_weight_matrix/hertz_to_mel_1/mul/x:output:02linear_to_mel_weight_matrix/hertz_to_mel_1/Log:y:0*
T0*
_output_shapes
: y
4linear_to_mel_weight_matrix/hertz_to_mel_2/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  /D�
2linear_to_mel_weight_matrix/hertz_to_mel_2/truedivRealDiv5linear_to_mel_weight_matrix/upper_edge_hertz:output:0=linear_to_mel_weight_matrix/hertz_to_mel_2/truediv/y:output:0*
T0*
_output_shapes
: u
0linear_to_mel_weight_matrix/hertz_to_mel_2/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
.linear_to_mel_weight_matrix/hertz_to_mel_2/addAddV29linear_to_mel_weight_matrix/hertz_to_mel_2/add/x:output:06linear_to_mel_weight_matrix/hertz_to_mel_2/truediv:z:0*
T0*
_output_shapes
: �
.linear_to_mel_weight_matrix/hertz_to_mel_2/LogLog2linear_to_mel_weight_matrix/hertz_to_mel_2/add:z:0*
T0*
_output_shapes
: u
0linear_to_mel_weight_matrix/hertz_to_mel_2/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 * ��D�
.linear_to_mel_weight_matrix/hertz_to_mel_2/mulMul9linear_to_mel_weight_matrix/hertz_to_mel_2/mul/x:output:02linear_to_mel_weight_matrix/hertz_to_mel_2/Log:y:0*
T0*
_output_shapes
: l
*linear_to_mel_weight_matrix/linspace_1/numConst*
_output_shapes
: *
dtype0*
value	B :4�
+linear_to_mel_weight_matrix/linspace_1/CastCast3linear_to_mel_weight_matrix/linspace_1/num:output:0*

DstT0*

SrcT0*
_output_shapes
: �
-linear_to_mel_weight_matrix/linspace_1/Cast_1Cast/linear_to_mel_weight_matrix/linspace_1/Cast:y:0*

DstT0*

SrcT0*
_output_shapes
: o
,linear_to_mel_weight_matrix/linspace_1/ShapeConst*
_output_shapes
: *
dtype0*
valueB q
.linear_to_mel_weight_matrix/linspace_1/Shape_1Const*
_output_shapes
: *
dtype0*
valueB �
4linear_to_mel_weight_matrix/linspace_1/BroadcastArgsBroadcastArgs5linear_to_mel_weight_matrix/linspace_1/Shape:output:07linear_to_mel_weight_matrix/linspace_1/Shape_1:output:0*
_output_shapes
: �
2linear_to_mel_weight_matrix/linspace_1/BroadcastToBroadcastTo2linear_to_mel_weight_matrix/hertz_to_mel_1/mul:z:09linear_to_mel_weight_matrix/linspace_1/BroadcastArgs:r0:0*
T0*
_output_shapes
: �
4linear_to_mel_weight_matrix/linspace_1/BroadcastTo_1BroadcastTo2linear_to_mel_weight_matrix/hertz_to_mel_2/mul:z:09linear_to_mel_weight_matrix/linspace_1/BroadcastArgs:r0:0*
T0*
_output_shapes
: w
5linear_to_mel_weight_matrix/linspace_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : �
1linear_to_mel_weight_matrix/linspace_1/ExpandDims
ExpandDims;linear_to_mel_weight_matrix/linspace_1/BroadcastTo:output:0>linear_to_mel_weight_matrix/linspace_1/ExpandDims/dim:output:0*
T0*
_output_shapes
:y
7linear_to_mel_weight_matrix/linspace_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
3linear_to_mel_weight_matrix/linspace_1/ExpandDims_1
ExpandDims=linear_to_mel_weight_matrix/linspace_1/BroadcastTo_1:output:0@linear_to_mel_weight_matrix/linspace_1/ExpandDims_1/dim:output:0*
T0*
_output_shapes
:x
.linear_to_mel_weight_matrix/linspace_1/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:x
.linear_to_mel_weight_matrix/linspace_1/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:�
:linear_to_mel_weight_matrix/linspace_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
<linear_to_mel_weight_matrix/linspace_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
<linear_to_mel_weight_matrix/linspace_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
4linear_to_mel_weight_matrix/linspace_1/strided_sliceStridedSlice7linear_to_mel_weight_matrix/linspace_1/Shape_3:output:0Clinear_to_mel_weight_matrix/linspace_1/strided_slice/stack:output:0Elinear_to_mel_weight_matrix/linspace_1/strided_slice/stack_1:output:0Elinear_to_mel_weight_matrix/linspace_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
,linear_to_mel_weight_matrix/linspace_1/add/yConst*
_output_shapes
: *
dtype0*
value	B : �
*linear_to_mel_weight_matrix/linspace_1/addAddV2=linear_to_mel_weight_matrix/linspace_1/strided_slice:output:05linear_to_mel_weight_matrix/linspace_1/add/y:output:0*
T0*
_output_shapes
: {
9linear_to_mel_weight_matrix/linspace_1/SelectV2/conditionConst*
_output_shapes
: *
dtype0
*
value	B
 Zs
1linear_to_mel_weight_matrix/linspace_1/SelectV2/tConst*
_output_shapes
: *
dtype0*
value	B : �
/linear_to_mel_weight_matrix/linspace_1/SelectV2SelectV2Blinear_to_mel_weight_matrix/linspace_1/SelectV2/condition:output:0:linear_to_mel_weight_matrix/linspace_1/SelectV2/t:output:0.linear_to_mel_weight_matrix/linspace_1/add:z:0*
T0*
_output_shapes
: n
,linear_to_mel_weight_matrix/linspace_1/sub/yConst*
_output_shapes
: *
dtype0*
value	B :�
*linear_to_mel_weight_matrix/linspace_1/subSub/linear_to_mel_weight_matrix/linspace_1/Cast:y:05linear_to_mel_weight_matrix/linspace_1/sub/y:output:0*
T0*
_output_shapes
: r
0linear_to_mel_weight_matrix/linspace_1/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B : �
.linear_to_mel_weight_matrix/linspace_1/MaximumMaximum.linear_to_mel_weight_matrix/linspace_1/sub:z:09linear_to_mel_weight_matrix/linspace_1/Maximum/y:output:0*
T0*
_output_shapes
: p
.linear_to_mel_weight_matrix/linspace_1/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
,linear_to_mel_weight_matrix/linspace_1/sub_1Sub/linear_to_mel_weight_matrix/linspace_1/Cast:y:07linear_to_mel_weight_matrix/linspace_1/sub_1/y:output:0*
T0*
_output_shapes
: t
2linear_to_mel_weight_matrix/linspace_1/Maximum_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
0linear_to_mel_weight_matrix/linspace_1/Maximum_1Maximum0linear_to_mel_weight_matrix/linspace_1/sub_1:z:0;linear_to_mel_weight_matrix/linspace_1/Maximum_1/y:output:0*
T0*
_output_shapes
: �
,linear_to_mel_weight_matrix/linspace_1/sub_2Sub<linear_to_mel_weight_matrix/linspace_1/ExpandDims_1:output:0:linear_to_mel_weight_matrix/linspace_1/ExpandDims:output:0*
T0*
_output_shapes
:�
-linear_to_mel_weight_matrix/linspace_1/Cast_2Cast4linear_to_mel_weight_matrix/linspace_1/Maximum_1:z:0*

DstT0*

SrcT0*
_output_shapes
: �
.linear_to_mel_weight_matrix/linspace_1/truedivRealDiv0linear_to_mel_weight_matrix/linspace_1/sub_2:z:01linear_to_mel_weight_matrix/linspace_1/Cast_2:y:0*
T0*
_output_shapes
:w
5linear_to_mel_weight_matrix/linspace_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : �
3linear_to_mel_weight_matrix/linspace_1/GreaterEqualGreaterEqual/linear_to_mel_weight_matrix/linspace_1/Cast:y:0>linear_to_mel_weight_matrix/linspace_1/GreaterEqual/y:output:0*
T0*
_output_shapes
: ~
3linear_to_mel_weight_matrix/linspace_1/SelectV2_1/eConst*
_output_shapes
: *
dtype0*
valueB :
����������
1linear_to_mel_weight_matrix/linspace_1/SelectV2_1SelectV27linear_to_mel_weight_matrix/linspace_1/GreaterEqual:z:04linear_to_mel_weight_matrix/linspace_1/Maximum_1:z:0<linear_to_mel_weight_matrix/linspace_1/SelectV2_1/e:output:0*
T0*
_output_shapes
: t
2linear_to_mel_weight_matrix/linspace_1/range/startConst*
_output_shapes
: *
dtype0	*
value	B	 Rt
2linear_to_mel_weight_matrix/linspace_1/range/deltaConst*
_output_shapes
: *
dtype0	*
value	B	 R�
1linear_to_mel_weight_matrix/linspace_1/range/CastCast:linear_to_mel_weight_matrix/linspace_1/SelectV2_1:output:0*

DstT0	*

SrcT0*
_output_shapes
: �
,linear_to_mel_weight_matrix/linspace_1/rangeRange;linear_to_mel_weight_matrix/linspace_1/range/start:output:05linear_to_mel_weight_matrix/linspace_1/range/Cast:y:0;linear_to_mel_weight_matrix/linspace_1/range/delta:output:0*

Tidx0	*
_output_shapes
:2�
-linear_to_mel_weight_matrix/linspace_1/Cast_3Cast5linear_to_mel_weight_matrix/linspace_1/range:output:0*

DstT0*

SrcT0	*
_output_shapes
:2v
4linear_to_mel_weight_matrix/linspace_1/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : v
4linear_to_mel_weight_matrix/linspace_1/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
.linear_to_mel_weight_matrix/linspace_1/range_1Range=linear_to_mel_weight_matrix/linspace_1/range_1/start:output:0=linear_to_mel_weight_matrix/linspace_1/strided_slice:output:0=linear_to_mel_weight_matrix/linspace_1/range_1/delta:output:0*
_output_shapes
:�
,linear_to_mel_weight_matrix/linspace_1/EqualEqual8linear_to_mel_weight_matrix/linspace_1/SelectV2:output:07linear_to_mel_weight_matrix/linspace_1/range_1:output:0*
T0*
_output_shapes
:u
3linear_to_mel_weight_matrix/linspace_1/SelectV2_2/eConst*
_output_shapes
: *
dtype0*
value	B :�
1linear_to_mel_weight_matrix/linspace_1/SelectV2_2SelectV20linear_to_mel_weight_matrix/linspace_1/Equal:z:02linear_to_mel_weight_matrix/linspace_1/Maximum:z:0<linear_to_mel_weight_matrix/linspace_1/SelectV2_2/e:output:0*
T0*
_output_shapes
:�
.linear_to_mel_weight_matrix/linspace_1/ReshapeReshape1linear_to_mel_weight_matrix/linspace_1/Cast_3:y:0:linear_to_mel_weight_matrix/linspace_1/SelectV2_2:output:0*
T0*
_output_shapes
:2�
*linear_to_mel_weight_matrix/linspace_1/mulMul2linear_to_mel_weight_matrix/linspace_1/truediv:z:07linear_to_mel_weight_matrix/linspace_1/Reshape:output:0*
T0*
_output_shapes
:2�
,linear_to_mel_weight_matrix/linspace_1/add_1AddV2:linear_to_mel_weight_matrix/linspace_1/ExpandDims:output:0.linear_to_mel_weight_matrix/linspace_1/mul:z:0*
T0*
_output_shapes
:2�
-linear_to_mel_weight_matrix/linspace_1/concatConcatV2:linear_to_mel_weight_matrix/linspace_1/ExpandDims:output:00linear_to_mel_weight_matrix/linspace_1/add_1:z:0<linear_to_mel_weight_matrix/linspace_1/ExpandDims_1:output:08linear_to_mel_weight_matrix/linspace_1/SelectV2:output:0*
N*
T0*
_output_shapes
:4{
1linear_to_mel_weight_matrix/linspace_1/zeros_likeConst*
_output_shapes
:*
dtype0*
valueB: n
,linear_to_mel_weight_matrix/linspace_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :�
<linear_to_mel_weight_matrix/linspace_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
>linear_to_mel_weight_matrix/linspace_1/strided_slice_1/stack_1Pack8linear_to_mel_weight_matrix/linspace_1/SelectV2:output:0*
N*
T0*
_output_shapes
:�
>linear_to_mel_weight_matrix/linspace_1/strided_slice_1/stack_2Pack5linear_to_mel_weight_matrix/linspace_1/Const:output:0*
N*
T0*
_output_shapes
:�
6linear_to_mel_weight_matrix/linspace_1/strided_slice_1StridedSlice7linear_to_mel_weight_matrix/linspace_1/Shape_2:output:0Elinear_to_mel_weight_matrix/linspace_1/strided_slice_1/stack:output:0Glinear_to_mel_weight_matrix/linspace_1/strided_slice_1/stack_1:output:0Glinear_to_mel_weight_matrix/linspace_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: �
6linear_to_mel_weight_matrix/linspace_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:�
0linear_to_mel_weight_matrix/linspace_1/Reshape_1Reshape/linear_to_mel_weight_matrix/linspace_1/Cast:y:0?linear_to_mel_weight_matrix/linspace_1/Reshape_1/shape:output:0*
T0*
_output_shapes
:p
.linear_to_mel_weight_matrix/linspace_1/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :�
,linear_to_mel_weight_matrix/linspace_1/add_2AddV28linear_to_mel_weight_matrix/linspace_1/SelectV2:output:07linear_to_mel_weight_matrix/linspace_1/add_2/y:output:0*
T0*
_output_shapes
: p
.linear_to_mel_weight_matrix/linspace_1/Const_1Const*
_output_shapes
: *
dtype0*
value	B : p
.linear_to_mel_weight_matrix/linspace_1/Const_2Const*
_output_shapes
: *
dtype0*
value	B :�
<linear_to_mel_weight_matrix/linspace_1/strided_slice_2/stackPack0linear_to_mel_weight_matrix/linspace_1/add_2:z:0*
N*
T0*
_output_shapes
:�
>linear_to_mel_weight_matrix/linspace_1/strided_slice_2/stack_1Pack7linear_to_mel_weight_matrix/linspace_1/Const_1:output:0*
N*
T0*
_output_shapes
:�
>linear_to_mel_weight_matrix/linspace_1/strided_slice_2/stack_2Pack7linear_to_mel_weight_matrix/linspace_1/Const_2:output:0*
N*
T0*
_output_shapes
:�
6linear_to_mel_weight_matrix/linspace_1/strided_slice_2StridedSlice7linear_to_mel_weight_matrix/linspace_1/Shape_2:output:0Elinear_to_mel_weight_matrix/linspace_1/strided_slice_2/stack:output:0Glinear_to_mel_weight_matrix/linspace_1/strided_slice_2/stack_1:output:0Glinear_to_mel_weight_matrix/linspace_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_maskv
4linear_to_mel_weight_matrix/linspace_1/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
/linear_to_mel_weight_matrix/linspace_1/concat_1ConcatV2?linear_to_mel_weight_matrix/linspace_1/strided_slice_1:output:09linear_to_mel_weight_matrix/linspace_1/Reshape_1:output:0?linear_to_mel_weight_matrix/linspace_1/strided_slice_2:output:0=linear_to_mel_weight_matrix/linspace_1/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
,linear_to_mel_weight_matrix/linspace_1/SliceSlice6linear_to_mel_weight_matrix/linspace_1/concat:output:0:linear_to_mel_weight_matrix/linspace_1/zeros_like:output:08linear_to_mel_weight_matrix/linspace_1/concat_1:output:0*
Index0*
T0*
_output_shapes
:4p
.linear_to_mel_weight_matrix/frame/frame_lengthConst*
_output_shapes
: *
dtype0*
value	B :n
,linear_to_mel_weight_matrix/frame/frame_stepConst*
_output_shapes
: *
dtype0*
value	B :q
&linear_to_mel_weight_matrix/frame/axisConst*
_output_shapes
: *
dtype0*
valueB :
���������q
'linear_to_mel_weight_matrix/frame/ShapeConst*
_output_shapes
:*
dtype0*
valueB:4o
,linear_to_mel_weight_matrix/frame/Size/ConstConst*
_output_shapes
: *
dtype0*
valueB h
&linear_to_mel_weight_matrix/frame/SizeConst*
_output_shapes
: *
dtype0*
value	B : q
.linear_to_mel_weight_matrix/frame/Size_1/ConstConst*
_output_shapes
: *
dtype0*
valueB j
(linear_to_mel_weight_matrix/frame/Size_1Const*
_output_shapes
: *
dtype0*
value	B : i
'linear_to_mel_weight_matrix/frame/ConstConst*
_output_shapes
: *
dtype0*
value	B : i
'linear_to_mel_weight_matrix/frame/sub/xConst*
_output_shapes
: *
dtype0*
value	B :4�
%linear_to_mel_weight_matrix/frame/subSub0linear_to_mel_weight_matrix/frame/sub/x:output:07linear_to_mel_weight_matrix/frame/frame_length:output:0*
T0*
_output_shapes
: �
*linear_to_mel_weight_matrix/frame/floordivFloorDiv)linear_to_mel_weight_matrix/frame/sub:z:05linear_to_mel_weight_matrix/frame/frame_step:output:0*
T0*
_output_shapes
: i
'linear_to_mel_weight_matrix/frame/add/xConst*
_output_shapes
: *
dtype0*
value	B :�
%linear_to_mel_weight_matrix/frame/addAddV20linear_to_mel_weight_matrix/frame/add/x:output:0.linear_to_mel_weight_matrix/frame/floordiv:z:0*
T0*
_output_shapes
: �
)linear_to_mel_weight_matrix/frame/MaximumMaximum0linear_to_mel_weight_matrix/frame/Const:output:0)linear_to_mel_weight_matrix/frame/add:z:0*
T0*
_output_shapes
: m
+linear_to_mel_weight_matrix/frame/gcd/ConstConst*
_output_shapes
: *
dtype0*
value	B :p
.linear_to_mel_weight_matrix/frame/floordiv_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
,linear_to_mel_weight_matrix/frame/floordiv_1FloorDiv7linear_to_mel_weight_matrix/frame/frame_length:output:07linear_to_mel_weight_matrix/frame/floordiv_1/y:output:0*
T0*
_output_shapes
: p
.linear_to_mel_weight_matrix/frame/floordiv_2/yConst*
_output_shapes
: *
dtype0*
value	B :�
,linear_to_mel_weight_matrix/frame/floordiv_2FloorDiv5linear_to_mel_weight_matrix/frame/frame_step:output:07linear_to_mel_weight_matrix/frame/floordiv_2/y:output:0*
T0*
_output_shapes
: t
1linear_to_mel_weight_matrix/frame/concat/values_0Const*
_output_shapes
: *
dtype0*
valueB {
1linear_to_mel_weight_matrix/frame/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:4t
1linear_to_mel_weight_matrix/frame/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB o
-linear_to_mel_weight_matrix/frame/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
(linear_to_mel_weight_matrix/frame/concatConcatV2:linear_to_mel_weight_matrix/frame/concat/values_0:output:0:linear_to_mel_weight_matrix/frame/concat/values_1:output:0:linear_to_mel_weight_matrix/frame/concat/values_2:output:06linear_to_mel_weight_matrix/frame/concat/axis:output:0*
N*
T0*
_output_shapes
:v
3linear_to_mel_weight_matrix/frame/concat_1/values_0Const*
_output_shapes
: *
dtype0*
valueB �
3linear_to_mel_weight_matrix/frame/concat_1/values_1Const*
_output_shapes
:*
dtype0*
valueB"4      v
3linear_to_mel_weight_matrix/frame/concat_1/values_2Const*
_output_shapes
: *
dtype0*
valueB q
/linear_to_mel_weight_matrix/frame/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
*linear_to_mel_weight_matrix/frame/concat_1ConcatV2<linear_to_mel_weight_matrix/frame/concat_1/values_0:output:0<linear_to_mel_weight_matrix/frame/concat_1/values_1:output:0<linear_to_mel_weight_matrix/frame/concat_1/values_2:output:08linear_to_mel_weight_matrix/frame/concat_1/axis:output:0*
N*
T0*
_output_shapes
:}
3linear_to_mel_weight_matrix/frame/zeros_like/tensorConst*
_output_shapes
:*
dtype0*
valueB:4v
,linear_to_mel_weight_matrix/frame/zeros_likeConst*
_output_shapes
:*
dtype0*
valueB: �
Alinear_to_mel_weight_matrix/frame/ones_like/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:s
1linear_to_mel_weight_matrix/frame/ones_like/ConstConst*
_output_shapes
: *
dtype0*
value	B :�
+linear_to_mel_weight_matrix/frame/ones_likeFillJlinear_to_mel_weight_matrix/frame/ones_like/Shape/shape_as_tensor:output:0:linear_to_mel_weight_matrix/frame/ones_like/Const:output:0*
T0*
_output_shapes
:�
.linear_to_mel_weight_matrix/frame/StridedSliceStridedSlice5linear_to_mel_weight_matrix/linspace_1/Slice:output:05linear_to_mel_weight_matrix/frame/zeros_like:output:01linear_to_mel_weight_matrix/frame/concat:output:04linear_to_mel_weight_matrix/frame/ones_like:output:0*
Index0*
T0*
_output_shapes
:4�
)linear_to_mel_weight_matrix/frame/ReshapeReshape7linear_to_mel_weight_matrix/frame/StridedSlice:output:03linear_to_mel_weight_matrix/frame/concat_1:output:0*
T0*
_output_shapes

:4o
-linear_to_mel_weight_matrix/frame/range/startConst*
_output_shapes
: *
dtype0*
value	B : o
-linear_to_mel_weight_matrix/frame/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
'linear_to_mel_weight_matrix/frame/rangeRange6linear_to_mel_weight_matrix/frame/range/start:output:0-linear_to_mel_weight_matrix/frame/Maximum:z:06linear_to_mel_weight_matrix/frame/range/delta:output:0*
_output_shapes
:2�
%linear_to_mel_weight_matrix/frame/mulMul0linear_to_mel_weight_matrix/frame/range:output:00linear_to_mel_weight_matrix/frame/floordiv_2:z:0*
T0*
_output_shapes
:2u
3linear_to_mel_weight_matrix/frame/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :�
1linear_to_mel_weight_matrix/frame/Reshape_1/shapePack-linear_to_mel_weight_matrix/frame/Maximum:z:0<linear_to_mel_weight_matrix/frame/Reshape_1/shape/1:output:0*
N*
T0*
_output_shapes
:�
+linear_to_mel_weight_matrix/frame/Reshape_1Reshape)linear_to_mel_weight_matrix/frame/mul:z:0:linear_to_mel_weight_matrix/frame/Reshape_1/shape:output:0*
T0*
_output_shapes

:2q
/linear_to_mel_weight_matrix/frame/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : q
/linear_to_mel_weight_matrix/frame/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
)linear_to_mel_weight_matrix/frame/range_1Range8linear_to_mel_weight_matrix/frame/range_1/start:output:00linear_to_mel_weight_matrix/frame/floordiv_1:z:08linear_to_mel_weight_matrix/frame/range_1/delta:output:0*
_output_shapes
:u
3linear_to_mel_weight_matrix/frame/Reshape_2/shape/0Const*
_output_shapes
: *
dtype0*
value	B :�
1linear_to_mel_weight_matrix/frame/Reshape_2/shapePack<linear_to_mel_weight_matrix/frame/Reshape_2/shape/0:output:00linear_to_mel_weight_matrix/frame/floordiv_1:z:0*
N*
T0*
_output_shapes
:�
+linear_to_mel_weight_matrix/frame/Reshape_2Reshape2linear_to_mel_weight_matrix/frame/range_1:output:0:linear_to_mel_weight_matrix/frame/Reshape_2/shape:output:0*
T0*
_output_shapes

:�
'linear_to_mel_weight_matrix/frame/add_1AddV24linear_to_mel_weight_matrix/frame/Reshape_1:output:04linear_to_mel_weight_matrix/frame/Reshape_2:output:0*
T0*
_output_shapes

:2l
)linear_to_mel_weight_matrix/frame/Const_1Const*
_output_shapes
: *
dtype0*
valueB l
)linear_to_mel_weight_matrix/frame/Const_2Const*
_output_shapes
: *
dtype0*
valueB �
(linear_to_mel_weight_matrix/frame/packedPack-linear_to_mel_weight_matrix/frame/Maximum:z:07linear_to_mel_weight_matrix/frame/frame_length:output:0*
N*
T0*
_output_shapes
:q
/linear_to_mel_weight_matrix/frame/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
*linear_to_mel_weight_matrix/frame/GatherV2GatherV22linear_to_mel_weight_matrix/frame/Reshape:output:0+linear_to_mel_weight_matrix/frame/add_1:z:08linear_to_mel_weight_matrix/frame/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*"
_output_shapes
:2q
/linear_to_mel_weight_matrix/frame/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
*linear_to_mel_weight_matrix/frame/concat_2ConcatV22linear_to_mel_weight_matrix/frame/Const_1:output:01linear_to_mel_weight_matrix/frame/packed:output:02linear_to_mel_weight_matrix/frame/Const_2:output:08linear_to_mel_weight_matrix/frame/concat_2/axis:output:0*
N*
T0*
_output_shapes
:�
+linear_to_mel_weight_matrix/frame/Reshape_3Reshape3linear_to_mel_weight_matrix/frame/GatherV2:output:03linear_to_mel_weight_matrix/frame/concat_2:output:0*
T0*
_output_shapes

:2m
+linear_to_mel_weight_matrix/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
!linear_to_mel_weight_matrix/splitSplit4linear_to_mel_weight_matrix/split/split_dim:output:04linear_to_mel_weight_matrix/frame/Reshape_3:output:0*
T0*2
_output_shapes 
:2:2:2*
	num_splitz
)linear_to_mel_weight_matrix/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   2   �
#linear_to_mel_weight_matrix/ReshapeReshape*linear_to_mel_weight_matrix/split:output:02linear_to_mel_weight_matrix/Reshape/shape:output:0*
T0*
_output_shapes

:2|
+linear_to_mel_weight_matrix/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   2   �
%linear_to_mel_weight_matrix/Reshape_1Reshape*linear_to_mel_weight_matrix/split:output:14linear_to_mel_weight_matrix/Reshape_1/shape:output:0*
T0*
_output_shapes

:2|
+linear_to_mel_weight_matrix/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"   2   �
%linear_to_mel_weight_matrix/Reshape_2Reshape*linear_to_mel_weight_matrix/split:output:24linear_to_mel_weight_matrix/Reshape_2/shape:output:0*
T0*
_output_shapes

:2�
linear_to_mel_weight_matrix/subSub/linear_to_mel_weight_matrix/ExpandDims:output:0,linear_to_mel_weight_matrix/Reshape:output:0*
T0*
_output_shapes
:	�2�
!linear_to_mel_weight_matrix/sub_1Sub.linear_to_mel_weight_matrix/Reshape_1:output:0,linear_to_mel_weight_matrix/Reshape:output:0*
T0*
_output_shapes

:2�
%linear_to_mel_weight_matrix/truediv_1RealDiv#linear_to_mel_weight_matrix/sub:z:0%linear_to_mel_weight_matrix/sub_1:z:0*
T0*
_output_shapes
:	�2�
!linear_to_mel_weight_matrix/sub_2Sub.linear_to_mel_weight_matrix/Reshape_2:output:0/linear_to_mel_weight_matrix/ExpandDims:output:0*
T0*
_output_shapes
:	�2�
!linear_to_mel_weight_matrix/sub_3Sub.linear_to_mel_weight_matrix/Reshape_2:output:0.linear_to_mel_weight_matrix/Reshape_1:output:0*
T0*
_output_shapes

:2�
%linear_to_mel_weight_matrix/truediv_2RealDiv%linear_to_mel_weight_matrix/sub_2:z:0%linear_to_mel_weight_matrix/sub_3:z:0*
T0*
_output_shapes
:	�2�
#linear_to_mel_weight_matrix/MinimumMinimum)linear_to_mel_weight_matrix/truediv_1:z:0)linear_to_mel_weight_matrix/truediv_2:z:0*
T0*
_output_shapes
:	�2�
#linear_to_mel_weight_matrix/MaximumMaximum*linear_to_mel_weight_matrix/Const:output:0'linear_to_mel_weight_matrix/Minimum:z:0*
T0*
_output_shapes
:	�2�
$linear_to_mel_weight_matrix/paddingsConst*
_output_shapes

:*
dtype0*)
value B"               �
linear_to_mel_weight_matrixPad'linear_to_mel_weight_matrix/Maximum:z:0-linear_to_mel_weight_matrix/paddings:output:0*
T0*
_output_shapes
:	�2q
MatMulMatMulAbs:y:0$linear_to_mel_weight_matrix:output:0*
T0*'
_output_shapes
:���������2J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5`
addAddV2MatMul:product:0add/y:output:0*
T0*'
_output_shapes
:���������2E
LogLogadd:z:0*
T0*'
_output_shapes
:���������2Y
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������p

ExpandDims
ExpandDimsLog:y:0ExpandDims/dim:output:0*
T0*+
_output_shapes
:���������2R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
ExpandDims_1
ExpandDimsExpandDims:output:0ExpandDims_1/dim:output:0*
T0*/
_output_shapes
:���������2�
StatefulPartitionedCallStatefulPartitionedCallExpandDims_1:output:0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37*3
Tin,
*2(*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*G
_read_only_resource_inputs)
'%	
 !"#$%&'*-
config_proto

CPU

GPU 2J 8� *3
f.R,
*__inference_signature_wrapper___call___753[
ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
���������{
ArgMaxArgMax StatefulPartitionedCall:output:0ArgMax/dimension:output:0*
T0*#
_output_shapes
:����������
GatherV2/paramsConst*
_output_shapes
:*
dtype0*[
valueRBPB
backgroundBdownBgoBleftBnoBoffBonBrightBstopBupByesBunknownO
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
GatherV2GatherV2GatherV2/params:output:0ArgMax:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:���������Z
IdentityIdentityArgMax:output:0^NoOp*
T0	*#
_output_shapes
:���������^

Identity_1IdentityGatherV2:output:0^NoOp*
T0*#
_output_shapes
:���������q

Identity_2Identity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:����������}:b2:b2: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:$' 

_user_specified_name3520:$& 

_user_specified_name3518:$% 

_user_specified_name3516:$$ 

_user_specified_name3514:$# 

_user_specified_name3512:$" 

_user_specified_name3510:$! 

_user_specified_name3508:$  

_user_specified_name3506:$ 

_user_specified_name3504:$ 

_user_specified_name3502:$ 

_user_specified_name3500:$ 

_user_specified_name3498:$ 

_user_specified_name3496:$ 

_user_specified_name3494:$ 

_user_specified_name3492:$ 

_user_specified_name3490:$ 

_user_specified_name3488:$ 

_user_specified_name3486:$ 

_user_specified_name3484:$ 

_user_specified_name3482:$ 

_user_specified_name3480:$ 

_user_specified_name3478:$ 

_user_specified_name3476:$ 

_user_specified_name3474:$ 

_user_specified_name3472:$ 

_user_specified_name3470:$ 

_user_specified_name3468:$ 

_user_specified_name3466:$ 

_user_specified_name3464:$
 

_user_specified_name3462:$	 

_user_specified_name3460:$ 

_user_specified_name3458:$ 

_user_specified_name3456:$ 

_user_specified_name3454:$ 

_user_specified_name3452:$ 

_user_specified_name3450:$ 

_user_specified_name3448:LH
&
_output_shapes
:b2

_user_specified_name3446:LH
&
_output_shapes
:b2

_user_specified_name3444:O K
(
_output_shapes
:����������}

_user_specified_nameinput
ڐ
�&
__inference___call___665
imageinput_unnormalized&
"functional_1_1_imageinput__1_sub_y'
#functional_1_1_imageinput__1_sqrt_xV
<functional_1_1_conv_1__1_convolution_readvariableop_resource:F
8functional_1_1_conv_1__1_reshape_readvariableop_resource:H
:functional_1_1_batchnorm_1__1_cast_readvariableop_resource:J
<functional_1_1_batchnorm_1__1_cast_1_readvariableop_resource:J
<functional_1_1_batchnorm_1__1_cast_2_readvariableop_resource:J
<functional_1_1_batchnorm_1__1_cast_3_readvariableop_resource:V
<functional_1_1_conv_2__1_convolution_readvariableop_resource:F
8functional_1_1_conv_2__1_reshape_readvariableop_resource:H
:functional_1_1_batchnorm_2__1_cast_readvariableop_resource:J
<functional_1_1_batchnorm_2__1_cast_1_readvariableop_resource:J
<functional_1_1_batchnorm_2__1_cast_2_readvariableop_resource:J
<functional_1_1_batchnorm_2__1_cast_3_readvariableop_resource:V
<functional_1_1_conv_3__1_convolution_readvariableop_resource:F
8functional_1_1_conv_3__1_reshape_readvariableop_resource:H
:functional_1_1_batchnorm_3__1_cast_readvariableop_resource:J
<functional_1_1_batchnorm_3__1_cast_1_readvariableop_resource:J
<functional_1_1_batchnorm_3__1_cast_2_readvariableop_resource:J
<functional_1_1_batchnorm_3__1_cast_3_readvariableop_resource:V
<functional_1_1_conv_4__1_convolution_readvariableop_resource: F
8functional_1_1_conv_4__1_reshape_readvariableop_resource: H
:functional_1_1_batchnorm_4__1_cast_readvariableop_resource: J
<functional_1_1_batchnorm_4__1_cast_1_readvariableop_resource: J
<functional_1_1_batchnorm_4__1_cast_2_readvariableop_resource: J
<functional_1_1_batchnorm_4__1_cast_3_readvariableop_resource: V
<functional_1_1_conv_5__1_convolution_readvariableop_resource: PF
8functional_1_1_conv_5__1_reshape_readvariableop_resource:PH
:functional_1_1_batchnorm_5__1_cast_readvariableop_resource:PJ
<functional_1_1_batchnorm_5__1_cast_1_readvariableop_resource:PJ
<functional_1_1_batchnorm_5__1_cast_2_readvariableop_resource:PJ
<functional_1_1_batchnorm_5__1_cast_3_readvariableop_resource:PS
?functional_1_1_lstm__1_lstm_cell_1_cast_readvariableop_resource:
��U
Afunctional_1_1_lstm__1_lstm_cell_1_cast_1_readvariableop_resource:
��O
@functional_1_1_lstm__1_lstm_cell_1_add_1_readvariableop_resource:	�F
3functional_1_1_fc_1__1_cast_readvariableop_resource:	�@
2functional_1_1_fc_1__1_add_readvariableop_resource:E
3functional_1_1_fc_2__1_cast_readvariableop_resource:@
2functional_1_1_fc_2__1_add_readvariableop_resource:
identity��1functional_1_1/batchnorm_1__1/Cast/ReadVariableOp�3functional_1_1/batchnorm_1__1/Cast_1/ReadVariableOp�3functional_1_1/batchnorm_1__1/Cast_2/ReadVariableOp�3functional_1_1/batchnorm_1__1/Cast_3/ReadVariableOp�1functional_1_1/batchnorm_2__1/Cast/ReadVariableOp�3functional_1_1/batchnorm_2__1/Cast_1/ReadVariableOp�3functional_1_1/batchnorm_2__1/Cast_2/ReadVariableOp�3functional_1_1/batchnorm_2__1/Cast_3/ReadVariableOp�1functional_1_1/batchnorm_3__1/Cast/ReadVariableOp�3functional_1_1/batchnorm_3__1/Cast_1/ReadVariableOp�3functional_1_1/batchnorm_3__1/Cast_2/ReadVariableOp�3functional_1_1/batchnorm_3__1/Cast_3/ReadVariableOp�1functional_1_1/batchnorm_4__1/Cast/ReadVariableOp�3functional_1_1/batchnorm_4__1/Cast_1/ReadVariableOp�3functional_1_1/batchnorm_4__1/Cast_2/ReadVariableOp�3functional_1_1/batchnorm_4__1/Cast_3/ReadVariableOp�1functional_1_1/batchnorm_5__1/Cast/ReadVariableOp�3functional_1_1/batchnorm_5__1/Cast_1/ReadVariableOp�3functional_1_1/batchnorm_5__1/Cast_2/ReadVariableOp�3functional_1_1/batchnorm_5__1/Cast_3/ReadVariableOp�/functional_1_1/conv_1__1/Reshape/ReadVariableOp�3functional_1_1/conv_1__1/convolution/ReadVariableOp�/functional_1_1/conv_2__1/Reshape/ReadVariableOp�3functional_1_1/conv_2__1/convolution/ReadVariableOp�/functional_1_1/conv_3__1/Reshape/ReadVariableOp�3functional_1_1/conv_3__1/convolution/ReadVariableOp�/functional_1_1/conv_4__1/Reshape/ReadVariableOp�3functional_1_1/conv_4__1/convolution/ReadVariableOp�/functional_1_1/conv_5__1/Reshape/ReadVariableOp�3functional_1_1/conv_5__1/convolution/ReadVariableOp�)functional_1_1/fc_1__1/Add/ReadVariableOp�*functional_1_1/fc_1__1/Cast/ReadVariableOp�)functional_1_1/fc_2__1/Add/ReadVariableOp�*functional_1_1/fc_2__1/Cast/ReadVariableOp�6functional_1_1/lstm__1/lstm_cell_1/Cast/ReadVariableOp�8functional_1_1/lstm__1/lstm_cell_1/Cast_1/ReadVariableOp�7functional_1_1/lstm__1/lstm_cell_1/add_1/ReadVariableOp�functional_1_1/lstm__1/while�
 functional_1_1/imageinput__1/SubSubimageinput_unnormalized"functional_1_1_imageinput__1_sub_y*
T0*/
_output_shapes
:���������b2
!functional_1_1/imageinput__1/SqrtSqrt#functional_1_1_imageinput__1_sqrt_x*
T0*&
_output_shapes
:b2g
"functional_1_1/imageinput__1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
$functional_1_1/imageinput__1/MaximumMaximum%functional_1_1/imageinput__1/Sqrt:y:0+functional_1_1/imageinput__1/Const:output:0*
T0*&
_output_shapes
:b2�
$functional_1_1/imageinput__1/truedivRealDiv$functional_1_1/imageinput__1/Sub:z:0(functional_1_1/imageinput__1/Maximum:z:0*
T0*/
_output_shapes
:���������b2�
%functional_1_1/zero_padding2d_1/ConstConst*
_output_shapes

:*
dtype0*9
value0B."                             �
#functional_1_1/zero_padding2d_1/PadPad(functional_1_1/imageinput__1/truediv:z:0.functional_1_1/zero_padding2d_1/Const:output:0*
T0*/
_output_shapes
:���������d4�
3functional_1_1/conv_1__1/convolution/ReadVariableOpReadVariableOp<functional_1_1_conv_1__1_convolution_readvariableop_resource*&
_output_shapes
:*
dtype0�
$functional_1_1/conv_1__1/convolutionConv2D,functional_1_1/zero_padding2d_1/Pad:output:0;functional_1_1/conv_1__1/convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������b2*
paddingVALID*
strides
�
/functional_1_1/conv_1__1/Reshape/ReadVariableOpReadVariableOp8functional_1_1_conv_1__1_reshape_readvariableop_resource*
_output_shapes
:*
dtype0
&functional_1_1/conv_1__1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            �
 functional_1_1/conv_1__1/ReshapeReshape7functional_1_1/conv_1__1/Reshape/ReadVariableOp:value:0/functional_1_1/conv_1__1/Reshape/shape:output:0*
T0*&
_output_shapes
:�
functional_1_1/conv_1__1/addAddV2-functional_1_1/conv_1__1/convolution:output:0)functional_1_1/conv_1__1/Reshape:output:0*
T0*/
_output_shapes
:���������b2�
1functional_1_1/batchnorm_1__1/Cast/ReadVariableOpReadVariableOp:functional_1_1_batchnorm_1__1_cast_readvariableop_resource*
_output_shapes
:*
dtype0�
3functional_1_1/batchnorm_1__1/Cast_1/ReadVariableOpReadVariableOp<functional_1_1_batchnorm_1__1_cast_1_readvariableop_resource*
_output_shapes
:*
dtype0�
3functional_1_1/batchnorm_1__1/Cast_2/ReadVariableOpReadVariableOp<functional_1_1_batchnorm_1__1_cast_2_readvariableop_resource*
_output_shapes
:*
dtype0�
3functional_1_1/batchnorm_1__1/Cast_3/ReadVariableOpReadVariableOp<functional_1_1_batchnorm_1__1_cast_3_readvariableop_resource*
_output_shapes
:*
dtype0r
-functional_1_1/batchnorm_1__1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
+functional_1_1/batchnorm_1__1/batchnorm/addAddV2;functional_1_1/batchnorm_1__1/Cast_1/ReadVariableOp:value:06functional_1_1/batchnorm_1__1/batchnorm/add/y:output:0*
T0*
_output_shapes
:�
-functional_1_1/batchnorm_1__1/batchnorm/RsqrtRsqrt/functional_1_1/batchnorm_1__1/batchnorm/add:z:0*
T0*
_output_shapes
:�
+functional_1_1/batchnorm_1__1/batchnorm/mulMul1functional_1_1/batchnorm_1__1/batchnorm/Rsqrt:y:0;functional_1_1/batchnorm_1__1/Cast_2/ReadVariableOp:value:0*
T0*
_output_shapes
:�
-functional_1_1/batchnorm_1__1/batchnorm/mul_1Mul functional_1_1/conv_1__1/add:z:0/functional_1_1/batchnorm_1__1/batchnorm/mul:z:0*
T0*/
_output_shapes
:���������b2�
-functional_1_1/batchnorm_1__1/batchnorm/mul_2Mul9functional_1_1/batchnorm_1__1/Cast/ReadVariableOp:value:0/functional_1_1/batchnorm_1__1/batchnorm/mul:z:0*
T0*
_output_shapes
:�
+functional_1_1/batchnorm_1__1/batchnorm/subSub;functional_1_1/batchnorm_1__1/Cast_3/ReadVariableOp:value:01functional_1_1/batchnorm_1__1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
-functional_1_1/batchnorm_1__1/batchnorm/add_1AddV21functional_1_1/batchnorm_1__1/batchnorm/mul_1:z:0/functional_1_1/batchnorm_1__1/batchnorm/sub:z:0*
T0*/
_output_shapes
:���������b2�
functional_1_1/re_lu_1/ReluRelu1functional_1_1/batchnorm_1__1/batchnorm/add_1:z:0*
T0*/
_output_shapes
:���������b2�
(functional_1_1/max_pooling2d_1/MaxPool2dMaxPool)functional_1_1/re_lu_1/Relu:activations:0*/
_output_shapes
:���������1*
ksize
*
paddingSAME*
strides
�
'functional_1_1/zero_padding2d_1_2/ConstConst*
_output_shapes

:*
dtype0*9
value0B."                             �
%functional_1_1/zero_padding2d_1_2/PadPad1functional_1_1/max_pooling2d_1/MaxPool2d:output:00functional_1_1/zero_padding2d_1_2/Const:output:0*
T0*/
_output_shapes
:���������3�
3functional_1_1/conv_2__1/convolution/ReadVariableOpReadVariableOp<functional_1_1_conv_2__1_convolution_readvariableop_resource*&
_output_shapes
:*
dtype0�
$functional_1_1/conv_2__1/convolutionConv2D.functional_1_1/zero_padding2d_1_2/Pad:output:0;functional_1_1/conv_2__1/convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������1*
paddingVALID*
strides
�
/functional_1_1/conv_2__1/Reshape/ReadVariableOpReadVariableOp8functional_1_1_conv_2__1_reshape_readvariableop_resource*
_output_shapes
:*
dtype0
&functional_1_1/conv_2__1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            �
 functional_1_1/conv_2__1/ReshapeReshape7functional_1_1/conv_2__1/Reshape/ReadVariableOp:value:0/functional_1_1/conv_2__1/Reshape/shape:output:0*
T0*&
_output_shapes
:�
functional_1_1/conv_2__1/addAddV2-functional_1_1/conv_2__1/convolution:output:0)functional_1_1/conv_2__1/Reshape:output:0*
T0*/
_output_shapes
:���������1�
1functional_1_1/batchnorm_2__1/Cast/ReadVariableOpReadVariableOp:functional_1_1_batchnorm_2__1_cast_readvariableop_resource*
_output_shapes
:*
dtype0�
3functional_1_1/batchnorm_2__1/Cast_1/ReadVariableOpReadVariableOp<functional_1_1_batchnorm_2__1_cast_1_readvariableop_resource*
_output_shapes
:*
dtype0�
3functional_1_1/batchnorm_2__1/Cast_2/ReadVariableOpReadVariableOp<functional_1_1_batchnorm_2__1_cast_2_readvariableop_resource*
_output_shapes
:*
dtype0�
3functional_1_1/batchnorm_2__1/Cast_3/ReadVariableOpReadVariableOp<functional_1_1_batchnorm_2__1_cast_3_readvariableop_resource*
_output_shapes
:*
dtype0r
-functional_1_1/batchnorm_2__1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
+functional_1_1/batchnorm_2__1/batchnorm/addAddV2;functional_1_1/batchnorm_2__1/Cast_1/ReadVariableOp:value:06functional_1_1/batchnorm_2__1/batchnorm/add/y:output:0*
T0*
_output_shapes
:�
-functional_1_1/batchnorm_2__1/batchnorm/RsqrtRsqrt/functional_1_1/batchnorm_2__1/batchnorm/add:z:0*
T0*
_output_shapes
:�
+functional_1_1/batchnorm_2__1/batchnorm/mulMul1functional_1_1/batchnorm_2__1/batchnorm/Rsqrt:y:0;functional_1_1/batchnorm_2__1/Cast_2/ReadVariableOp:value:0*
T0*
_output_shapes
:�
-functional_1_1/batchnorm_2__1/batchnorm/mul_1Mul functional_1_1/conv_2__1/add:z:0/functional_1_1/batchnorm_2__1/batchnorm/mul:z:0*
T0*/
_output_shapes
:���������1�
-functional_1_1/batchnorm_2__1/batchnorm/mul_2Mul9functional_1_1/batchnorm_2__1/Cast/ReadVariableOp:value:0/functional_1_1/batchnorm_2__1/batchnorm/mul:z:0*
T0*
_output_shapes
:�
+functional_1_1/batchnorm_2__1/batchnorm/subSub;functional_1_1/batchnorm_2__1/Cast_3/ReadVariableOp:value:01functional_1_1/batchnorm_2__1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
-functional_1_1/batchnorm_2__1/batchnorm/add_1AddV21functional_1_1/batchnorm_2__1/batchnorm/mul_1:z:0/functional_1_1/batchnorm_2__1/batchnorm/sub:z:0*
T0*/
_output_shapes
:���������1�
functional_1_1/re_lu_1_2/ReluRelu1functional_1_1/batchnorm_2__1/batchnorm/add_1:z:0*
T0*/
_output_shapes
:���������1�
*functional_1_1/max_pooling2d_1_2/MaxPool2dMaxPool+functional_1_1/re_lu_1_2/Relu:activations:0*/
_output_shapes
:���������*
ksize
*
paddingSAME*
strides
�
'functional_1_1/zero_padding2d_2_1/ConstConst*
_output_shapes

:*
dtype0*9
value0B."                             �
%functional_1_1/zero_padding2d_2_1/PadPad3functional_1_1/max_pooling2d_1_2/MaxPool2d:output:00functional_1_1/zero_padding2d_2_1/Const:output:0*
T0*/
_output_shapes
:����������
3functional_1_1/conv_3__1/convolution/ReadVariableOpReadVariableOp<functional_1_1_conv_3__1_convolution_readvariableop_resource*&
_output_shapes
:*
dtype0�
$functional_1_1/conv_3__1/convolutionConv2D.functional_1_1/zero_padding2d_2_1/Pad:output:0;functional_1_1/conv_3__1/convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
/functional_1_1/conv_3__1/Reshape/ReadVariableOpReadVariableOp8functional_1_1_conv_3__1_reshape_readvariableop_resource*
_output_shapes
:*
dtype0
&functional_1_1/conv_3__1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            �
 functional_1_1/conv_3__1/ReshapeReshape7functional_1_1/conv_3__1/Reshape/ReadVariableOp:value:0/functional_1_1/conv_3__1/Reshape/shape:output:0*
T0*&
_output_shapes
:�
functional_1_1/conv_3__1/addAddV2-functional_1_1/conv_3__1/convolution:output:0)functional_1_1/conv_3__1/Reshape:output:0*
T0*/
_output_shapes
:����������
1functional_1_1/batchnorm_3__1/Cast/ReadVariableOpReadVariableOp:functional_1_1_batchnorm_3__1_cast_readvariableop_resource*
_output_shapes
:*
dtype0�
3functional_1_1/batchnorm_3__1/Cast_1/ReadVariableOpReadVariableOp<functional_1_1_batchnorm_3__1_cast_1_readvariableop_resource*
_output_shapes
:*
dtype0�
3functional_1_1/batchnorm_3__1/Cast_2/ReadVariableOpReadVariableOp<functional_1_1_batchnorm_3__1_cast_2_readvariableop_resource*
_output_shapes
:*
dtype0�
3functional_1_1/batchnorm_3__1/Cast_3/ReadVariableOpReadVariableOp<functional_1_1_batchnorm_3__1_cast_3_readvariableop_resource*
_output_shapes
:*
dtype0r
-functional_1_1/batchnorm_3__1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
+functional_1_1/batchnorm_3__1/batchnorm/addAddV2;functional_1_1/batchnorm_3__1/Cast_1/ReadVariableOp:value:06functional_1_1/batchnorm_3__1/batchnorm/add/y:output:0*
T0*
_output_shapes
:�
-functional_1_1/batchnorm_3__1/batchnorm/RsqrtRsqrt/functional_1_1/batchnorm_3__1/batchnorm/add:z:0*
T0*
_output_shapes
:�
+functional_1_1/batchnorm_3__1/batchnorm/mulMul1functional_1_1/batchnorm_3__1/batchnorm/Rsqrt:y:0;functional_1_1/batchnorm_3__1/Cast_2/ReadVariableOp:value:0*
T0*
_output_shapes
:�
-functional_1_1/batchnorm_3__1/batchnorm/mul_1Mul functional_1_1/conv_3__1/add:z:0/functional_1_1/batchnorm_3__1/batchnorm/mul:z:0*
T0*/
_output_shapes
:����������
-functional_1_1/batchnorm_3__1/batchnorm/mul_2Mul9functional_1_1/batchnorm_3__1/Cast/ReadVariableOp:value:0/functional_1_1/batchnorm_3__1/batchnorm/mul:z:0*
T0*
_output_shapes
:�
+functional_1_1/batchnorm_3__1/batchnorm/subSub;functional_1_1/batchnorm_3__1/Cast_3/ReadVariableOp:value:01functional_1_1/batchnorm_3__1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
-functional_1_1/batchnorm_3__1/batchnorm/add_1AddV21functional_1_1/batchnorm_3__1/batchnorm/mul_1:z:0/functional_1_1/batchnorm_3__1/batchnorm/sub:z:0*
T0*/
_output_shapes
:����������
functional_1_1/re_lu_2_1/ReluRelu1functional_1_1/batchnorm_3__1/batchnorm/add_1:z:0*
T0*/
_output_shapes
:����������
*functional_1_1/max_pooling2d_2_1/MaxPool2dMaxPool+functional_1_1/re_lu_2_1/Relu:activations:0*/
_output_shapes
:���������*
ksize
*
paddingSAME*
strides
�
'functional_1_1/zero_padding2d_3_1/ConstConst*
_output_shapes

:*
dtype0*9
value0B."                             �
%functional_1_1/zero_padding2d_3_1/PadPad3functional_1_1/max_pooling2d_2_1/MaxPool2d:output:00functional_1_1/zero_padding2d_3_1/Const:output:0*
T0*/
_output_shapes
:���������	�
3functional_1_1/conv_4__1/convolution/ReadVariableOpReadVariableOp<functional_1_1_conv_4__1_convolution_readvariableop_resource*&
_output_shapes
: *
dtype0�
$functional_1_1/conv_4__1/convolutionConv2D.functional_1_1/zero_padding2d_3_1/Pad:output:0;functional_1_1/conv_4__1/convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
�
/functional_1_1/conv_4__1/Reshape/ReadVariableOpReadVariableOp8functional_1_1_conv_4__1_reshape_readvariableop_resource*
_output_shapes
: *
dtype0
&functional_1_1/conv_4__1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             �
 functional_1_1/conv_4__1/ReshapeReshape7functional_1_1/conv_4__1/Reshape/ReadVariableOp:value:0/functional_1_1/conv_4__1/Reshape/shape:output:0*
T0*&
_output_shapes
: �
functional_1_1/conv_4__1/addAddV2-functional_1_1/conv_4__1/convolution:output:0)functional_1_1/conv_4__1/Reshape:output:0*
T0*/
_output_shapes
:��������� �
1functional_1_1/batchnorm_4__1/Cast/ReadVariableOpReadVariableOp:functional_1_1_batchnorm_4__1_cast_readvariableop_resource*
_output_shapes
: *
dtype0�
3functional_1_1/batchnorm_4__1/Cast_1/ReadVariableOpReadVariableOp<functional_1_1_batchnorm_4__1_cast_1_readvariableop_resource*
_output_shapes
: *
dtype0�
3functional_1_1/batchnorm_4__1/Cast_2/ReadVariableOpReadVariableOp<functional_1_1_batchnorm_4__1_cast_2_readvariableop_resource*
_output_shapes
: *
dtype0�
3functional_1_1/batchnorm_4__1/Cast_3/ReadVariableOpReadVariableOp<functional_1_1_batchnorm_4__1_cast_3_readvariableop_resource*
_output_shapes
: *
dtype0r
-functional_1_1/batchnorm_4__1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
+functional_1_1/batchnorm_4__1/batchnorm/addAddV2;functional_1_1/batchnorm_4__1/Cast_1/ReadVariableOp:value:06functional_1_1/batchnorm_4__1/batchnorm/add/y:output:0*
T0*
_output_shapes
: �
-functional_1_1/batchnorm_4__1/batchnorm/RsqrtRsqrt/functional_1_1/batchnorm_4__1/batchnorm/add:z:0*
T0*
_output_shapes
: �
+functional_1_1/batchnorm_4__1/batchnorm/mulMul1functional_1_1/batchnorm_4__1/batchnorm/Rsqrt:y:0;functional_1_1/batchnorm_4__1/Cast_2/ReadVariableOp:value:0*
T0*
_output_shapes
: �
-functional_1_1/batchnorm_4__1/batchnorm/mul_1Mul functional_1_1/conv_4__1/add:z:0/functional_1_1/batchnorm_4__1/batchnorm/mul:z:0*
T0*/
_output_shapes
:��������� �
-functional_1_1/batchnorm_4__1/batchnorm/mul_2Mul9functional_1_1/batchnorm_4__1/Cast/ReadVariableOp:value:0/functional_1_1/batchnorm_4__1/batchnorm/mul:z:0*
T0*
_output_shapes
: �
+functional_1_1/batchnorm_4__1/batchnorm/subSub;functional_1_1/batchnorm_4__1/Cast_3/ReadVariableOp:value:01functional_1_1/batchnorm_4__1/batchnorm/mul_2:z:0*
T0*
_output_shapes
: �
-functional_1_1/batchnorm_4__1/batchnorm/add_1AddV21functional_1_1/batchnorm_4__1/batchnorm/mul_1:z:0/functional_1_1/batchnorm_4__1/batchnorm/sub:z:0*
T0*/
_output_shapes
:��������� �
functional_1_1/re_lu_3_1/ReluRelu1functional_1_1/batchnorm_4__1/batchnorm/add_1:z:0*
T0*/
_output_shapes
:��������� �
'functional_1_1/zero_padding2d_4_1/ConstConst*
_output_shapes

:*
dtype0*9
value0B."                             �
%functional_1_1/zero_padding2d_4_1/PadPad+functional_1_1/re_lu_3_1/Relu:activations:00functional_1_1/zero_padding2d_4_1/Const:output:0*
T0*/
_output_shapes
:���������	 �
3functional_1_1/conv_5__1/convolution/ReadVariableOpReadVariableOp<functional_1_1_conv_5__1_convolution_readvariableop_resource*&
_output_shapes
: P*
dtype0�
$functional_1_1/conv_5__1/convolutionConv2D.functional_1_1/zero_padding2d_4_1/Pad:output:0;functional_1_1/conv_5__1/convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P*
paddingVALID*
strides
�
/functional_1_1/conv_5__1/Reshape/ReadVariableOpReadVariableOp8functional_1_1_conv_5__1_reshape_readvariableop_resource*
_output_shapes
:P*
dtype0
&functional_1_1/conv_5__1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         P   �
 functional_1_1/conv_5__1/ReshapeReshape7functional_1_1/conv_5__1/Reshape/ReadVariableOp:value:0/functional_1_1/conv_5__1/Reshape/shape:output:0*
T0*&
_output_shapes
:P�
functional_1_1/conv_5__1/addAddV2-functional_1_1/conv_5__1/convolution:output:0)functional_1_1/conv_5__1/Reshape:output:0*
T0*/
_output_shapes
:���������P�
1functional_1_1/batchnorm_5__1/Cast/ReadVariableOpReadVariableOp:functional_1_1_batchnorm_5__1_cast_readvariableop_resource*
_output_shapes
:P*
dtype0�
3functional_1_1/batchnorm_5__1/Cast_1/ReadVariableOpReadVariableOp<functional_1_1_batchnorm_5__1_cast_1_readvariableop_resource*
_output_shapes
:P*
dtype0�
3functional_1_1/batchnorm_5__1/Cast_2/ReadVariableOpReadVariableOp<functional_1_1_batchnorm_5__1_cast_2_readvariableop_resource*
_output_shapes
:P*
dtype0�
3functional_1_1/batchnorm_5__1/Cast_3/ReadVariableOpReadVariableOp<functional_1_1_batchnorm_5__1_cast_3_readvariableop_resource*
_output_shapes
:P*
dtype0r
-functional_1_1/batchnorm_5__1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
+functional_1_1/batchnorm_5__1/batchnorm/addAddV2;functional_1_1/batchnorm_5__1/Cast_1/ReadVariableOp:value:06functional_1_1/batchnorm_5__1/batchnorm/add/y:output:0*
T0*
_output_shapes
:P�
-functional_1_1/batchnorm_5__1/batchnorm/RsqrtRsqrt/functional_1_1/batchnorm_5__1/batchnorm/add:z:0*
T0*
_output_shapes
:P�
+functional_1_1/batchnorm_5__1/batchnorm/mulMul1functional_1_1/batchnorm_5__1/batchnorm/Rsqrt:y:0;functional_1_1/batchnorm_5__1/Cast_2/ReadVariableOp:value:0*
T0*
_output_shapes
:P�
-functional_1_1/batchnorm_5__1/batchnorm/mul_1Mul functional_1_1/conv_5__1/add:z:0/functional_1_1/batchnorm_5__1/batchnorm/mul:z:0*
T0*/
_output_shapes
:���������P�
-functional_1_1/batchnorm_5__1/batchnorm/mul_2Mul9functional_1_1/batchnorm_5__1/Cast/ReadVariableOp:value:0/functional_1_1/batchnorm_5__1/batchnorm/mul:z:0*
T0*
_output_shapes
:P�
+functional_1_1/batchnorm_5__1/batchnorm/subSub;functional_1_1/batchnorm_5__1/Cast_3/ReadVariableOp:value:01functional_1_1/batchnorm_5__1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:P�
-functional_1_1/batchnorm_5__1/batchnorm/add_1AddV21functional_1_1/batchnorm_5__1/batchnorm/mul_1:z:0/functional_1_1/batchnorm_5__1/batchnorm/sub:z:0*
T0*/
_output_shapes
:���������P�
functional_1_1/re_lu_4_1/ReluRelu1functional_1_1/batchnorm_5__1/batchnorm/add_1:z:0*
T0*/
_output_shapes
:���������P�
*functional_1_1/max_pooling2d_3_1/MaxPool2dMaxPool+functional_1_1/re_lu_4_1/Relu:activations:0*/
_output_shapes
:���������P*
ksize
*
paddingVALID*
strides
�
'functional_1_1/permute_1/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             �
"functional_1_1/permute_1/transpose	Transpose3functional_1_1/max_pooling2d_3_1/MaxPool2d:output:00functional_1_1/permute_1/transpose/perm:output:0*
T0*/
_output_shapes
:���������Pw
&functional_1_1/flatten_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����0  �
 functional_1_1/flatten_1/ReshapeReshape&functional_1_1/permute_1/transpose:y:0/functional_1_1/flatten_1/Reshape/shape:output:0*
T0*(
_output_shapes
:�����������
functional_1_1/reshape_1/ShapeShape)functional_1_1/flatten_1/Reshape:output:0*
T0*
_output_shapes
::��v
,functional_1_1/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.functional_1_1/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.functional_1_1/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
&functional_1_1/reshape_1/strided_sliceStridedSlice'functional_1_1/reshape_1/Shape:output:05functional_1_1/reshape_1/strided_slice/stack:output:07functional_1_1/reshape_1/strided_slice/stack_1:output:07functional_1_1/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(functional_1_1/reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :k
(functional_1_1/reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :��
&functional_1_1/reshape_1/Reshape/shapePack/functional_1_1/reshape_1/strided_slice:output:01functional_1_1/reshape_1/Reshape/shape/1:output:01functional_1_1/reshape_1/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:�
 functional_1_1/reshape_1/ReshapeReshape)functional_1_1/flatten_1/Reshape:output:0/functional_1_1/reshape_1/Reshape/shape:output:0*
T0*,
_output_shapes
:�����������
functional_1_1/lstm__1/ShapeShape)functional_1_1/reshape_1/Reshape:output:0*
T0*
_output_shapes
::��t
*functional_1_1/lstm__1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,functional_1_1/lstm__1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,functional_1_1/lstm__1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
$functional_1_1/lstm__1/strided_sliceStridedSlice%functional_1_1/lstm__1/Shape:output:03functional_1_1/lstm__1/strided_slice/stack:output:05functional_1_1/lstm__1/strided_slice/stack_1:output:05functional_1_1/lstm__1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
%functional_1_1/lstm__1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :��
#functional_1_1/lstm__1/zeros/packedPack-functional_1_1/lstm__1/strided_slice:output:0.functional_1_1/lstm__1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:g
"functional_1_1/lstm__1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
functional_1_1/lstm__1/zerosFill,functional_1_1/lstm__1/zeros/packed:output:0+functional_1_1/lstm__1/zeros/Const:output:0*
T0*(
_output_shapes
:����������j
'functional_1_1/lstm__1/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :��
%functional_1_1/lstm__1/zeros_1/packedPack-functional_1_1/lstm__1/strided_slice:output:00functional_1_1/lstm__1/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:i
$functional_1_1/lstm__1/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
functional_1_1/lstm__1/zeros_1Fill.functional_1_1/lstm__1/zeros_1/packed:output:0-functional_1_1/lstm__1/zeros_1/Const:output:0*
T0*(
_output_shapes
:�����������
,functional_1_1/lstm__1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            �
.functional_1_1/lstm__1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           �
.functional_1_1/lstm__1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
&functional_1_1/lstm__1/strided_slice_1StridedSlice)functional_1_1/reshape_1/Reshape:output:05functional_1_1/lstm__1/strided_slice_1/stack:output:07functional_1_1/lstm__1/strided_slice_1/stack_1:output:07functional_1_1/lstm__1/strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*

begin_mask*
end_mask*
shrink_axis_maskz
%functional_1_1/lstm__1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
 functional_1_1/lstm__1/transpose	Transpose)functional_1_1/reshape_1/Reshape:output:0.functional_1_1/lstm__1/transpose/perm:output:0*
T0*,
_output_shapes
:����������}
2functional_1_1/lstm__1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������s
1functional_1_1/lstm__1/TensorArrayV2/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
$functional_1_1/lstm__1/TensorArrayV2TensorListReserve;functional_1_1/lstm__1/TensorArrayV2/element_shape:output:0:functional_1_1/lstm__1/TensorArrayV2/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
Lfunctional_1_1/lstm__1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����0  �
>functional_1_1/lstm__1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor$functional_1_1/lstm__1/transpose:y:0Ufunctional_1_1/lstm__1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���v
,functional_1_1/lstm__1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.functional_1_1/lstm__1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.functional_1_1/lstm__1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
&functional_1_1/lstm__1/strided_slice_2StridedSlice$functional_1_1/lstm__1/transpose:y:05functional_1_1/lstm__1/strided_slice_2/stack:output:07functional_1_1/lstm__1/strided_slice_2/stack_1:output:07functional_1_1/lstm__1/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask�
6functional_1_1/lstm__1/lstm_cell_1/Cast/ReadVariableOpReadVariableOp?functional_1_1_lstm__1_lstm_cell_1_cast_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
)functional_1_1/lstm__1/lstm_cell_1/MatMulMatMul/functional_1_1/lstm__1/strided_slice_2:output:0>functional_1_1/lstm__1/lstm_cell_1/Cast/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
8functional_1_1/lstm__1/lstm_cell_1/Cast_1/ReadVariableOpReadVariableOpAfunctional_1_1_lstm__1_lstm_cell_1_cast_1_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
+functional_1_1/lstm__1/lstm_cell_1/MatMul_1MatMul%functional_1_1/lstm__1/zeros:output:0@functional_1_1/lstm__1/lstm_cell_1/Cast_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
&functional_1_1/lstm__1/lstm_cell_1/addAddV23functional_1_1/lstm__1/lstm_cell_1/MatMul:product:05functional_1_1/lstm__1/lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
7functional_1_1/lstm__1/lstm_cell_1/add_1/ReadVariableOpReadVariableOp@functional_1_1_lstm__1_lstm_cell_1_add_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
(functional_1_1/lstm__1/lstm_cell_1/add_1AddV2*functional_1_1/lstm__1/lstm_cell_1/add:z:0?functional_1_1/lstm__1/lstm_cell_1/add_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������t
2functional_1_1/lstm__1/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
(functional_1_1/lstm__1/lstm_cell_1/splitSplit;functional_1_1/lstm__1/lstm_cell_1/split/split_dim:output:0,functional_1_1/lstm__1/lstm_cell_1/add_1:z:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split�
*functional_1_1/lstm__1/lstm_cell_1/SigmoidSigmoid1functional_1_1/lstm__1/lstm_cell_1/split:output:0*
T0*(
_output_shapes
:�����������
,functional_1_1/lstm__1/lstm_cell_1/Sigmoid_1Sigmoid1functional_1_1/lstm__1/lstm_cell_1/split:output:1*
T0*(
_output_shapes
:�����������
&functional_1_1/lstm__1/lstm_cell_1/mulMul0functional_1_1/lstm__1/lstm_cell_1/Sigmoid_1:y:0'functional_1_1/lstm__1/zeros_1:output:0*
T0*(
_output_shapes
:�����������
'functional_1_1/lstm__1/lstm_cell_1/TanhTanh1functional_1_1/lstm__1/lstm_cell_1/split:output:2*
T0*(
_output_shapes
:�����������
(functional_1_1/lstm__1/lstm_cell_1/mul_1Mul.functional_1_1/lstm__1/lstm_cell_1/Sigmoid:y:0+functional_1_1/lstm__1/lstm_cell_1/Tanh:y:0*
T0*(
_output_shapes
:�����������
(functional_1_1/lstm__1/lstm_cell_1/add_2AddV2*functional_1_1/lstm__1/lstm_cell_1/mul:z:0,functional_1_1/lstm__1/lstm_cell_1/mul_1:z:0*
T0*(
_output_shapes
:�����������
,functional_1_1/lstm__1/lstm_cell_1/Sigmoid_2Sigmoid1functional_1_1/lstm__1/lstm_cell_1/split:output:3*
T0*(
_output_shapes
:�����������
)functional_1_1/lstm__1/lstm_cell_1/Tanh_1Tanh,functional_1_1/lstm__1/lstm_cell_1/add_2:z:0*
T0*(
_output_shapes
:�����������
(functional_1_1/lstm__1/lstm_cell_1/mul_2Mul0functional_1_1/lstm__1/lstm_cell_1/Sigmoid_2:y:0-functional_1_1/lstm__1/lstm_cell_1/Tanh_1:y:0*
T0*(
_output_shapes
:�����������
4functional_1_1/lstm__1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   u
3functional_1_1/lstm__1/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
&functional_1_1/lstm__1/TensorArrayV2_1TensorListReserve=functional_1_1/lstm__1/TensorArrayV2_1/element_shape:output:0<functional_1_1/lstm__1/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���]
functional_1_1/lstm__1/timeConst*
_output_shapes
: *
dtype0*
value	B : c
!functional_1_1/lstm__1/Rank/ConstConst*
_output_shapes
: *
dtype0*
value	B :]
functional_1_1/lstm__1/RankConst*
_output_shapes
: *
dtype0*
value	B : d
"functional_1_1/lstm__1/range/startConst*
_output_shapes
: *
dtype0*
value	B : d
"functional_1_1/lstm__1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
functional_1_1/lstm__1/rangeRange+functional_1_1/lstm__1/range/start:output:0$functional_1_1/lstm__1/Rank:output:0+functional_1_1/lstm__1/range/delta:output:0*
_output_shapes
: b
 functional_1_1/lstm__1/Max/inputConst*
_output_shapes
: *
dtype0*
value	B :�
functional_1_1/lstm__1/MaxMax)functional_1_1/lstm__1/Max/input:output:0%functional_1_1/lstm__1/range:output:0*
T0*
_output_shapes
: k
)functional_1_1/lstm__1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
functional_1_1/lstm__1/whileWhile2functional_1_1/lstm__1/while/loop_counter:output:0#functional_1_1/lstm__1/Max:output:0$functional_1_1/lstm__1/time:output:0/functional_1_1/lstm__1/TensorArrayV2_1:handle:0%functional_1_1/lstm__1/zeros:output:0'functional_1_1/lstm__1/zeros_1:output:0Nfunctional_1_1/lstm__1/TensorArrayUnstack/TensorListFromTensor:output_handle:0?functional_1_1_lstm__1_lstm_cell_1_cast_readvariableop_resourceAfunctional_1_1_lstm__1_lstm_cell_1_cast_1_readvariableop_resource@functional_1_1_lstm__1_lstm_cell_1_add_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*L
_output_shapes:
8: : : : :����������:����������: : : : *%
_read_only_resource_inputs
	*A
body9R7
5__inference_functional_1_1_lstm__1_while_body_945_397*A
cond9R7
5__inference_functional_1_1_lstm__1_while_cond_944_351*K
output_shapes:
8: : : : :����������:����������: : : : *
parallel_iterations �
Gfunctional_1_1/lstm__1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   �
9functional_1_1/lstm__1/TensorArrayV2Stack/TensorListStackTensorListStack%functional_1_1/lstm__1/while:output:3Pfunctional_1_1/lstm__1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:����������*
element_dtype0*
num_elements
,functional_1_1/lstm__1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������x
.functional_1_1/lstm__1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: x
.functional_1_1/lstm__1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
&functional_1_1/lstm__1/strided_slice_3StridedSliceBfunctional_1_1/lstm__1/TensorArrayV2Stack/TensorListStack:tensor:05functional_1_1/lstm__1/strided_slice_3/stack:output:07functional_1_1/lstm__1/strided_slice_3/stack_1:output:07functional_1_1/lstm__1/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask|
'functional_1_1/lstm__1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
"functional_1_1/lstm__1/transpose_1	TransposeBfunctional_1_1/lstm__1/TensorArrayV2Stack/TensorListStack:tensor:00functional_1_1/lstm__1/transpose_1/perm:output:0*
T0*,
_output_shapes
:�����������
 functional_1_1/reshape_1_2/ShapeShape&functional_1_1/lstm__1/transpose_1:y:0*
T0*
_output_shapes
::��x
.functional_1_1/reshape_1_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0functional_1_1/reshape_1_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0functional_1_1/reshape_1_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(functional_1_1/reshape_1_2/strided_sliceStridedSlice)functional_1_1/reshape_1_2/Shape:output:07functional_1_1/reshape_1_2/strided_slice/stack:output:09functional_1_1/reshape_1_2/strided_slice/stack_1:output:09functional_1_1/reshape_1_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
*functional_1_1/reshape_1_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :��
(functional_1_1/reshape_1_2/Reshape/shapePack1functional_1_1/reshape_1_2/strided_slice:output:03functional_1_1/reshape_1_2/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:�
"functional_1_1/reshape_1_2/ReshapeReshape&functional_1_1/lstm__1/transpose_1:y:01functional_1_1/reshape_1_2/Reshape/shape:output:0*
T0*(
_output_shapes
:�����������
*functional_1_1/fc_1__1/Cast/ReadVariableOpReadVariableOp3functional_1_1_fc_1__1_cast_readvariableop_resource*
_output_shapes
:	�*
dtype0�
functional_1_1/fc_1__1/MatMulMatMul+functional_1_1/reshape_1_2/Reshape:output:02functional_1_1/fc_1__1/Cast/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
)functional_1_1/fc_1__1/Add/ReadVariableOpReadVariableOp2functional_1_1_fc_1__1_add_readvariableop_resource*
_output_shapes
:*
dtype0�
functional_1_1/fc_1__1/AddAddV2'functional_1_1/fc_1__1/MatMul:product:01functional_1_1/fc_1__1/Add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������w
functional_1_1/re_lu_5_1/ReluRelufunctional_1_1/fc_1__1/Add:z:0*
T0*'
_output_shapes
:����������
*functional_1_1/fc_2__1/Cast/ReadVariableOpReadVariableOp3functional_1_1_fc_2__1_cast_readvariableop_resource*
_output_shapes

:*
dtype0�
functional_1_1/fc_2__1/MatMulMatMul+functional_1_1/re_lu_5_1/Relu:activations:02functional_1_1/fc_2__1/Cast/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
)functional_1_1/fc_2__1/Add/ReadVariableOpReadVariableOp2functional_1_1_fc_2__1_add_readvariableop_resource*
_output_shapes
:*
dtype0�
functional_1_1/fc_2__1/AddAddV2'functional_1_1/fc_2__1/MatMul:product:01functional_1_1/fc_2__1/Add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������w
functional_1_1/re_lu_6_1/ReluRelufunctional_1_1/fc_2__1/Add:z:0*
T0*'
_output_shapes
:����������
 functional_1_1/softmax_1/SoftmaxSoftmax+functional_1_1/re_lu_6_1/Relu:activations:0*
T0*'
_output_shapes
:����������
NoOpNoOp2^functional_1_1/batchnorm_1__1/Cast/ReadVariableOp4^functional_1_1/batchnorm_1__1/Cast_1/ReadVariableOp4^functional_1_1/batchnorm_1__1/Cast_2/ReadVariableOp4^functional_1_1/batchnorm_1__1/Cast_3/ReadVariableOp2^functional_1_1/batchnorm_2__1/Cast/ReadVariableOp4^functional_1_1/batchnorm_2__1/Cast_1/ReadVariableOp4^functional_1_1/batchnorm_2__1/Cast_2/ReadVariableOp4^functional_1_1/batchnorm_2__1/Cast_3/ReadVariableOp2^functional_1_1/batchnorm_3__1/Cast/ReadVariableOp4^functional_1_1/batchnorm_3__1/Cast_1/ReadVariableOp4^functional_1_1/batchnorm_3__1/Cast_2/ReadVariableOp4^functional_1_1/batchnorm_3__1/Cast_3/ReadVariableOp2^functional_1_1/batchnorm_4__1/Cast/ReadVariableOp4^functional_1_1/batchnorm_4__1/Cast_1/ReadVariableOp4^functional_1_1/batchnorm_4__1/Cast_2/ReadVariableOp4^functional_1_1/batchnorm_4__1/Cast_3/ReadVariableOp2^functional_1_1/batchnorm_5__1/Cast/ReadVariableOp4^functional_1_1/batchnorm_5__1/Cast_1/ReadVariableOp4^functional_1_1/batchnorm_5__1/Cast_2/ReadVariableOp4^functional_1_1/batchnorm_5__1/Cast_3/ReadVariableOp0^functional_1_1/conv_1__1/Reshape/ReadVariableOp4^functional_1_1/conv_1__1/convolution/ReadVariableOp0^functional_1_1/conv_2__1/Reshape/ReadVariableOp4^functional_1_1/conv_2__1/convolution/ReadVariableOp0^functional_1_1/conv_3__1/Reshape/ReadVariableOp4^functional_1_1/conv_3__1/convolution/ReadVariableOp0^functional_1_1/conv_4__1/Reshape/ReadVariableOp4^functional_1_1/conv_4__1/convolution/ReadVariableOp0^functional_1_1/conv_5__1/Reshape/ReadVariableOp4^functional_1_1/conv_5__1/convolution/ReadVariableOp*^functional_1_1/fc_1__1/Add/ReadVariableOp+^functional_1_1/fc_1__1/Cast/ReadVariableOp*^functional_1_1/fc_2__1/Add/ReadVariableOp+^functional_1_1/fc_2__1/Cast/ReadVariableOp7^functional_1_1/lstm__1/lstm_cell_1/Cast/ReadVariableOp9^functional_1_1/lstm__1/lstm_cell_1/Cast_1/ReadVariableOp8^functional_1_1/lstm__1/lstm_cell_1/add_1/ReadVariableOp^functional_1_1/lstm__1/while*
_output_shapes
 y
IdentityIdentity*functional_1_1/softmax_1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������b2:b2:b2: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2f
1functional_1_1/batchnorm_1__1/Cast/ReadVariableOp1functional_1_1/batchnorm_1__1/Cast/ReadVariableOp2j
3functional_1_1/batchnorm_1__1/Cast_1/ReadVariableOp3functional_1_1/batchnorm_1__1/Cast_1/ReadVariableOp2j
3functional_1_1/batchnorm_1__1/Cast_2/ReadVariableOp3functional_1_1/batchnorm_1__1/Cast_2/ReadVariableOp2j
3functional_1_1/batchnorm_1__1/Cast_3/ReadVariableOp3functional_1_1/batchnorm_1__1/Cast_3/ReadVariableOp2f
1functional_1_1/batchnorm_2__1/Cast/ReadVariableOp1functional_1_1/batchnorm_2__1/Cast/ReadVariableOp2j
3functional_1_1/batchnorm_2__1/Cast_1/ReadVariableOp3functional_1_1/batchnorm_2__1/Cast_1/ReadVariableOp2j
3functional_1_1/batchnorm_2__1/Cast_2/ReadVariableOp3functional_1_1/batchnorm_2__1/Cast_2/ReadVariableOp2j
3functional_1_1/batchnorm_2__1/Cast_3/ReadVariableOp3functional_1_1/batchnorm_2__1/Cast_3/ReadVariableOp2f
1functional_1_1/batchnorm_3__1/Cast/ReadVariableOp1functional_1_1/batchnorm_3__1/Cast/ReadVariableOp2j
3functional_1_1/batchnorm_3__1/Cast_1/ReadVariableOp3functional_1_1/batchnorm_3__1/Cast_1/ReadVariableOp2j
3functional_1_1/batchnorm_3__1/Cast_2/ReadVariableOp3functional_1_1/batchnorm_3__1/Cast_2/ReadVariableOp2j
3functional_1_1/batchnorm_3__1/Cast_3/ReadVariableOp3functional_1_1/batchnorm_3__1/Cast_3/ReadVariableOp2f
1functional_1_1/batchnorm_4__1/Cast/ReadVariableOp1functional_1_1/batchnorm_4__1/Cast/ReadVariableOp2j
3functional_1_1/batchnorm_4__1/Cast_1/ReadVariableOp3functional_1_1/batchnorm_4__1/Cast_1/ReadVariableOp2j
3functional_1_1/batchnorm_4__1/Cast_2/ReadVariableOp3functional_1_1/batchnorm_4__1/Cast_2/ReadVariableOp2j
3functional_1_1/batchnorm_4__1/Cast_3/ReadVariableOp3functional_1_1/batchnorm_4__1/Cast_3/ReadVariableOp2f
1functional_1_1/batchnorm_5__1/Cast/ReadVariableOp1functional_1_1/batchnorm_5__1/Cast/ReadVariableOp2j
3functional_1_1/batchnorm_5__1/Cast_1/ReadVariableOp3functional_1_1/batchnorm_5__1/Cast_1/ReadVariableOp2j
3functional_1_1/batchnorm_5__1/Cast_2/ReadVariableOp3functional_1_1/batchnorm_5__1/Cast_2/ReadVariableOp2j
3functional_1_1/batchnorm_5__1/Cast_3/ReadVariableOp3functional_1_1/batchnorm_5__1/Cast_3/ReadVariableOp2b
/functional_1_1/conv_1__1/Reshape/ReadVariableOp/functional_1_1/conv_1__1/Reshape/ReadVariableOp2j
3functional_1_1/conv_1__1/convolution/ReadVariableOp3functional_1_1/conv_1__1/convolution/ReadVariableOp2b
/functional_1_1/conv_2__1/Reshape/ReadVariableOp/functional_1_1/conv_2__1/Reshape/ReadVariableOp2j
3functional_1_1/conv_2__1/convolution/ReadVariableOp3functional_1_1/conv_2__1/convolution/ReadVariableOp2b
/functional_1_1/conv_3__1/Reshape/ReadVariableOp/functional_1_1/conv_3__1/Reshape/ReadVariableOp2j
3functional_1_1/conv_3__1/convolution/ReadVariableOp3functional_1_1/conv_3__1/convolution/ReadVariableOp2b
/functional_1_1/conv_4__1/Reshape/ReadVariableOp/functional_1_1/conv_4__1/Reshape/ReadVariableOp2j
3functional_1_1/conv_4__1/convolution/ReadVariableOp3functional_1_1/conv_4__1/convolution/ReadVariableOp2b
/functional_1_1/conv_5__1/Reshape/ReadVariableOp/functional_1_1/conv_5__1/Reshape/ReadVariableOp2j
3functional_1_1/conv_5__1/convolution/ReadVariableOp3functional_1_1/conv_5__1/convolution/ReadVariableOp2V
)functional_1_1/fc_1__1/Add/ReadVariableOp)functional_1_1/fc_1__1/Add/ReadVariableOp2X
*functional_1_1/fc_1__1/Cast/ReadVariableOp*functional_1_1/fc_1__1/Cast/ReadVariableOp2V
)functional_1_1/fc_2__1/Add/ReadVariableOp)functional_1_1/fc_2__1/Add/ReadVariableOp2X
*functional_1_1/fc_2__1/Cast/ReadVariableOp*functional_1_1/fc_2__1/Cast/ReadVariableOp2p
6functional_1_1/lstm__1/lstm_cell_1/Cast/ReadVariableOp6functional_1_1/lstm__1/lstm_cell_1/Cast/ReadVariableOp2t
8functional_1_1/lstm__1/lstm_cell_1/Cast_1/ReadVariableOp8functional_1_1/lstm__1/lstm_cell_1/Cast_1/ReadVariableOp2r
7functional_1_1/lstm__1/lstm_cell_1/add_1/ReadVariableOp7functional_1_1/lstm__1/lstm_cell_1/add_1/ReadVariableOp2<
functional_1_1/lstm__1/whilefunctional_1_1/lstm__1/while:('$
"
_user_specified_name
resource:(&$
"
_user_specified_name
resource:(%$
"
_user_specified_name
resource:($$
"
_user_specified_name
resource:(#$
"
_user_specified_name
resource:("$
"
_user_specified_name
resource:(!$
"
_user_specified_name
resource:( $
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:IE
&
_output_shapes
:b2

_user_specified_namex:IE
&
_output_shapes
:b2

_user_specified_namey:h d
/
_output_shapes
:���������b2
1
_user_specified_nameimageinput_unnormalized
� 
�	
*__inference_signature_wrapper___call___709
imageinput_unnormalized
unknown
	unknown_0#
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:$

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:$

unknown_19: 

unknown_20: 

unknown_21: 

unknown_22: 

unknown_23: 

unknown_24: $

unknown_25: P

unknown_26:P

unknown_27:P

unknown_28:P

unknown_29:P

unknown_30:P

unknown_31:
��

unknown_32:
��

unknown_33:	�

unknown_34:	�

unknown_35:

unknown_36:

unknown_37:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallimageinput_unnormalizedunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37*3
Tin,
*2(*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*G
_read_only_resource_inputs)
'%	
 !"#$%&'*-
config_proto

CPU

GPU 2J 8� *!
fR
__inference___call___665<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������b2:b2:b2: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:$' 

_user_specified_name1130:$& 

_user_specified_name1128:$% 

_user_specified_name1126:$$ 

_user_specified_name1124:$# 

_user_specified_name1122:$" 

_user_specified_name1120:$! 

_user_specified_name1118:$  

_user_specified_name1116:$ 

_user_specified_name1114:$ 

_user_specified_name1112:$ 

_user_specified_name1110:$ 

_user_specified_name1108:$ 

_user_specified_name1106:$ 

_user_specified_name1104:$ 

_user_specified_name1102:$ 

_user_specified_name1100:$ 

_user_specified_name1098:$ 

_user_specified_name1096:$ 

_user_specified_name1094:$ 

_user_specified_name1092:$ 

_user_specified_name1090:$ 

_user_specified_name1088:$ 

_user_specified_name1086:$ 

_user_specified_name1084:$ 

_user_specified_name1082:$ 

_user_specified_name1080:$ 

_user_specified_name1078:$ 

_user_specified_name1076:$ 

_user_specified_name1074:$
 

_user_specified_name1072:$	 

_user_specified_name1070:$ 

_user_specified_name1068:$ 

_user_specified_name1066:$ 

_user_specified_name1064:$ 

_user_specified_name1062:$ 

_user_specified_name1060:$ 

_user_specified_name1058:LH
&
_output_shapes
:b2

_user_specified_name1056:LH
&
_output_shapes
:b2

_user_specified_name1054:h d
/
_output_shapes
:���������b2
1
_user_specified_nameimageinput_unnormalized"�L
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
&
input
serving_default_input:0 9
	class_ids,
StatefulPartitionedCall:0	���������;
class_names,
StatefulPartitionedCall:1���������?
predictions0
StatefulPartitionedCall:2���������tensorflow/serving/predict*�
serving_float�
6
input-
serving_float_input:0����������};
	class_ids.
StatefulPartitionedCall_1:0	���������=
class_names.
StatefulPartitionedCall_1:1���������A
predictions2
StatefulPartitionedCall_1:2���������tensorflow/serving/predict:�L
t
	model
__call__
call_float_input
call_string_input

signatures"
_generic_user_object
�
	variables
trainable_variables
non_trainable_variables
	_all_variables

_misc_assets

signatures
#_self_saveable_object_factories
	serve"
_generic_user_object
�
trace_0
trace_12�
__inference___call___3047
__inference___call___3531�
���
FullArgSpec
args�	
jinput
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 ztrace_0ztrace_1
�
	capture_0
	capture_1B�
__inference___call___2397input"�
���
FullArgSpec
args�	
jinput
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z	capture_0z	capture_1
�
	capture_0
	capture_1B�
__inference___call___1913input"�
���
FullArgSpec
args�	
jinput
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z	capture_0z	capture_1
?
serving_default
serving_float"
signature_map
�
0
1
2
3
4
5
6
7
8
9
10
11
 12
!13
"14
#15
$16
%17
&18
'19
(20
)21
*22
+23
,24
-25
.26
/27
028
129
230
331
432
533
634
735
836
937
:38
;39
<40
=41"
trackable_list_wrapper
�
0
1
2
3
4
5
6
 7
#8
$9
%10
&11
)12
*13
+14
,15
/16
017
118
219
520
621
722
923
:24
<25
=26"
trackable_list_wrapper
�
0
1
2
3
4
!5
"6
'7
(8
-9
.10
311
412
813
;14"
trackable_list_wrapper
�
>0
?1
@2
A3
B4
C5
D6
E7
F8
G9
H10
I11
J12
K13
L14
M15
N16
O17
P18
Q19
R20
S21
T22
U23
V24
W25
X26
Y27
Z28
[29
\30
]31
^32
_33
`34
a35
b36"
trackable_list_wrapper
 "
trackable_list_wrapper
7
	cserve
dserving_default"
signature_map
 "
trackable_dict_wrapper
�
etrace_02�
__inference___call___665�
���
FullArgSpec
args�

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *>�;
9�6
imageinput_unnormalized���������b2zetrace_0
�
	capture_0
	capture_1B�
__inference___call___3047input"�
���
FullArgSpec
args�	
jinput
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z	capture_0z	capture_1
�
	capture_0
	capture_1B�
__inference___call___3531input"�
���
FullArgSpec
args�	
jinput
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z	capture_0z	capture_1
!J	
Const_1jtf.TrackableConstant
J
Constjtf.TrackableConstant
�
	capture_0
	capture_1B�
"__inference_signature_wrapper_2485input"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs�	
jinput
kwonlydefaults
 
annotations� *
 z	capture_0z	capture_1
�
	capture_0
	capture_1B�
"__inference_signature_wrapper_2573input"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs�	
jinput
kwonlydefaults
 
annotations� *
 z	capture_0z	capture_1
$:"b22imageinput_/mean
(:&b22imageinput_/variance
:	 2imageinput_/count
(:&2conv_1_/kernel
:2conv_1_/bias
 :2batchnorm_1_/gamma
:2batchnorm_1_/beta
$:"2batchnorm_1_/moving_mean
(:&2batchnorm_1_/moving_variance
(:&2conv_2_/kernel
:2conv_2_/bias
 :2batchnorm_2_/gamma
:2batchnorm_2_/beta
$:"2batchnorm_2_/moving_mean
(:&2batchnorm_2_/moving_variance
(:&2conv_3_/kernel
:2conv_3_/bias
 :2batchnorm_3_/gamma
:2batchnorm_3_/beta
$:"2batchnorm_3_/moving_mean
(:&2batchnorm_3_/moving_variance
(:& 2conv_4_/kernel
: 2conv_4_/bias
 : 2batchnorm_4_/gamma
: 2batchnorm_4_/beta
$:" 2batchnorm_4_/moving_mean
(:& 2batchnorm_4_/moving_variance
(:& P2conv_5_/kernel
:P2conv_5_/bias
 :P2batchnorm_5_/gamma
:P2batchnorm_5_/beta
$:"P2batchnorm_5_/moving_mean
(:&P2batchnorm_5_/moving_variance
*:(
��2lstm_/lstm_cell/kernel
4:2
��2 lstm_/lstm_cell/recurrent_kernel
#:!�2lstm_/lstm_cell/bias
 :2seed_generator_state
:	�2fc_1_/kernel
:2
fc_1_/bias
 :2seed_generator_state
:2fc_2_/kernel
:2
fc_2_/bias
:2batchnorm_1_/beta
(:&2conv_2_/kernel
:2batchnorm_3_/beta
(:& 2conv_4_/kernel
:P2conv_5_/bias
(:&2conv_1_/kernel
:2conv_3_/bias
(:& P2conv_5_/kernel
(:&2conv_3_/kernel
*:(
��2lstm_/lstm_cell/kernel
:	�2fc_1_/kernel
:2conv_1_/bias
 :2batchnorm_2_/gamma
 : 2batchnorm_4_/gamma
 :P2batchnorm_5_/gamma
4:2
��2 lstm_/lstm_cell/recurrent_kernel
:2fc_2_/kernel
#:!�2lstm_/lstm_cell/bias
:2
fc_1_/bias
 :2batchnorm_3_/gamma
:2
fc_2_/bias
:2batchnorm_2_/beta
: 2batchnorm_4_/beta
:P2batchnorm_5_/beta
 :2batchnorm_1_/gamma
:2conv_2_/bias
: 2conv_4_/bias
(:&P2batchnorm_5_/moving_variance
(:&2batchnorm_2_/moving_variance
(:&2batchnorm_1_/moving_variance
$:"2batchnorm_2_/moving_mean
$:" 2batchnorm_4_/moving_mean
$:"P2batchnorm_5_/moving_mean
(:& 2batchnorm_4_/moving_variance
$:"2batchnorm_1_/moving_mean
$:"2batchnorm_3_/moving_mean
(:&2batchnorm_3_/moving_variance
�
	capture_0
	capture_1B�
*__inference_signature_wrapper___call___709imageinput_unnormalized"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 ,

kwonlyargs�
jimageinput_unnormalized
kwonlydefaults
 
annotations� *
 z	capture_0z	capture_1
�
	capture_0
	capture_1B�
*__inference_signature_wrapper___call___753imageinput_unnormalized"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 ,

kwonlyargs�
jimageinput_unnormalized
kwonlydefaults
 
annotations� *
 z	capture_0z	capture_1
�
	capture_0
	capture_1B�
__inference___call___665imageinput_unnormalized"�
���
FullArgSpec
args�

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z	capture_0z	capture_1�
__inference___call___1913�'!" #$'(%&)*-.+,/034125679:<=�
�
�
input 
� "���
,
	class_ids�
	class_ids���������	
0
class_names!�
class_names���������
4
predictions%�"
predictions����������
__inference___call___2397�'!" #$'(%&)*-.+,/034125679:<=/�,
%�"
 �
input����������}
� "���
,
	class_ids�
	class_ids���������	
0
class_names!�
class_names���������
4
predictions%�"
predictions����������
__inference___call___3047�'!" #$'(%&)*-.+,/034125679:<=�
�
�
input 
� "���
,
	class_ids�
	class_ids���������	
0
class_names!�
class_names���������
4
predictions%�"
predictions����������
__inference___call___3531�'!" #$'(%&)*-.+,/034125679:<=/�,
%�"
 �
input����������}
� "���
,
	class_ids�
	class_ids���������	
0
class_names!�
class_names���������
4
predictions%�"
predictions����������
__inference___call___665�'!" #$'(%&)*-.+,/034125679:<=H�E
>�;
9�6
imageinput_unnormalized���������b2
� "!�
unknown����������
"__inference_signature_wrapper_2485�'!" #$'(%&)*-.+,/034125679:<=&�#
� 
�

input�
input "���
,
	class_ids�
	class_ids���������	
0
class_names!�
class_names���������
4
predictions%�"
predictions����������
"__inference_signature_wrapper_2573�'!" #$'(%&)*-.+,/034125679:<=8�5
� 
.�+
)
input �
input����������}"���
,
	class_ids�
	class_ids���������	
0
class_names!�
class_names���������
4
predictions%�"
predictions����������
*__inference_signature_wrapper___call___709�'!" #$'(%&)*-.+,/034125679:<=c�`
� 
Y�V
T
imageinput_unnormalized9�6
imageinput_unnormalized���������b2"3�0
.
output_0"�
output_0����������
*__inference_signature_wrapper___call___753�'!" #$'(%&)*-.+,/034125679:<=c�`
� 
Y�V
T
imageinput_unnormalized9�6
imageinput_unnormalized���������b2"3�0
.
output_0"�
output_0���������