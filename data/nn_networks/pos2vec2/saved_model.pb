��	
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
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
$
DisableCopyOnRead
resource�
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
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( ""
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype�
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
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
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
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.14.02v2.14.0-rc1-21-g4dacf3f368e8��
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
r
dense_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_20/bias
k
!dense_20/bias/Read/ReadVariableOpReadVariableOpdense_20/bias*
_output_shapes
:d*
dtype0
{
dense_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�d* 
shared_namedense_20/kernel
t
#dense_20/kernel/Read/ReadVariableOpReadVariableOpdense_20/kernel*
_output_shapes
:	�d*
dtype0
s
dense_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_19/bias
l
!dense_19/bias/Read/ReadVariableOpReadVariableOpdense_19/bias*
_output_shapes	
:�*
dtype0
|
dense_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��* 
shared_namedense_19/kernel
u
#dense_19/kernel/Read/ReadVariableOpReadVariableOpdense_19/kernel* 
_output_shapes
:
��*
dtype0
s
dense_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_18/bias
l
!dense_18/bias/Read/ReadVariableOpReadVariableOpdense_18/bias*
_output_shapes	
:�*
dtype0
|
dense_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��* 
shared_namedense_18/kernel
u
#dense_18/kernel/Read/ReadVariableOpReadVariableOpdense_18/kernel* 
_output_shapes
:
��*
dtype0
s
dense_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_17/bias
l
!dense_17/bias/Read/ReadVariableOpReadVariableOpdense_17/bias*
_output_shapes	
:�*
dtype0
|
dense_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��* 
shared_namedense_17/kernel
u
#dense_17/kernel/Read/ReadVariableOpReadVariableOpdense_17/kernel* 
_output_shapes
:
��*
dtype0
s
dense_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_16/bias
l
!dense_16/bias/Read/ReadVariableOpReadVariableOpdense_16/bias*
_output_shapes	
:�*
dtype0
|
dense_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��* 
shared_namedense_16/kernel
u
#dense_16/kernel/Read/ReadVariableOpReadVariableOpdense_16/kernel* 
_output_shapes
:
��*
dtype0
�
serving_default_input_5Placeholder*5
_output_shapes#
!:�������������������*
dtype0**
shape!:�������������������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_5dense_16/kerneldense_16/biasdense_17/kerneldense_17/biasdense_18/kerneldense_18/biasdense_19/kerneldense_19/biasdense_20/kerneldense_20/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������d*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference_signature_wrapper_10212

NoOpNoOp
�1
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�0
value�0B�0 B�0
�
layer-0
layer_with_weights-0
layer-1
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
	_default_save_signature

	optimizer

signatures*
* 
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
J
0
1
2
3
4
5
6
7
8
 9*
J
0
1
2
3
4
5
6
7
8
 9*
* 
�
!non_trainable_variables

"layers
#metrics
$layer_regularization_losses
%layer_metrics
	variables
trainable_variables
regularization_losses
__call__
	_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

&trace_0
'trace_1* 

(trace_0
)trace_1* 
* 
O
*
_variables
+_iterations
,_learning_rate
-_update_step_xla*

.serving_default* 
�
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses

kernel
bias*
�
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses

kernel
bias*
�
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
*@&call_and_return_all_conditional_losses

kernel
bias*
�
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses

kernel
bias*
�
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses

kernel
 bias*
J
0
1
2
3
4
5
6
7
8
 9*
J
0
1
2
3
4
5
6
7
8
 9*
* 
�
Mnon_trainable_variables

Nlayers
Ometrics
Player_regularization_losses
Qlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Rtrace_0
Strace_1* 

Ttrace_0
Utrace_1* 
OI
VARIABLE_VALUEdense_16/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_16/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_17/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_17/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_18/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_18/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_19/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_19/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_20/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_20/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1*

V0
W1*
* 
* 
* 
* 
* 
* 

+0*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 

0
1*

0
1*
* 
�
Xnon_trainable_variables

Ylayers
Zmetrics
[layer_regularization_losses
\layer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses*

]trace_0* 

^trace_0* 

0
1*

0
1*
* 
�
_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses*

dtrace_0* 

etrace_0* 

0
1*

0
1*
* 
�
fnon_trainable_variables

glayers
hmetrics
ilayer_regularization_losses
jlayer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses*

ktrace_0* 

ltrace_0* 

0
1*

0
1*
* 
�
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses*

rtrace_0* 

strace_0* 

0
 1*

0
 1*
* 
�
tnon_trainable_variables

ulayers
vmetrics
wlayer_regularization_losses
xlayer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses*

ytrace_0* 

ztrace_0* 
* 
'
0
1
2
3
4*
* 
* 
* 
* 
* 
* 
* 
8
{	variables
|	keras_api
	}total
	~count*
L
	variables
�	keras_api

�total

�count
�
_fn_kwargs*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

}0
~1*

{	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamedense_16/kerneldense_16/biasdense_17/kerneldense_17/biasdense_18/kerneldense_18/biasdense_19/kerneldense_19/biasdense_20/kerneldense_20/bias	iterationlearning_ratetotal_1count_1totalcountConst*
Tin
2*
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
GPU 2J 8� *'
f"R 
__inference__traced_save_10530
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_16/kerneldense_16/biasdense_17/kerneldense_17/biasdense_18/kerneldense_18/biasdense_19/kerneldense_19/biasdense_20/kerneldense_20/bias	iterationlearning_ratetotal_1count_1totalcount*
Tin
2*
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
GPU 2J 8� **
f%R#
!__inference__traced_restore_10587��
�
�
(__inference_dense_16_layer_call_fn_10221

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_16_layer_call_and_return_conditional_losses_9809}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:�������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:�������������������: : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name10217:%!

_user_specified_name10215:] Y
5
_output_shapes#
!:�������������������
 
_user_specified_nameinputs
�
�
B__inference_dense_16_layer_call_and_return_conditional_losses_9809

inputs5
!tensordot_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp|
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*5
_output_shapes#
!:��������������������
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*5
_output_shapes#
!:�������������������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:�������������������^
ReluReluBiasAdd:output:0*
T0*5
_output_shapes#
!:�������������������o
IdentityIdentityRelu:activations:0^NoOp*
T0*5
_output_shapes#
!:�������������������V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:�������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:] Y
5
_output_shapes#
!:�������������������
 
_user_specified_nameinputs
�
�
C__inference_dense_20_layer_call_and_return_conditional_losses_10412

inputs4
!tensordot_readvariableop_resource:	�d-
biasadd_readvariableop_resource:d
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	�d*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*5
_output_shapes#
!:��������������������
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:dY
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :������������������dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :������������������d]
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :������������������dn
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :������������������dV
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:�������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:] Y
5
_output_shapes#
!:�������������������
 
_user_specified_nameinputs
�
�
#__inference_signature_wrapper_10212
input_5
unknown:
��
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:	�d
	unknown_8:d
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������d*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *(
f#R!
__inference__wrapped_model_9776|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������d<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:�������������������: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%
!

_user_specified_name10208:%	!

_user_specified_name10206:%!

_user_specified_name10204:%!

_user_specified_name10202:%!

_user_specified_name10200:%!

_user_specified_name10198:%!

_user_specified_name10196:%!

_user_specified_name10194:%!

_user_specified_name10192:%!

_user_specified_name10190:^ Z
5
_output_shapes#
!:�������������������
!
_user_specified_name	input_5
�
�
B__inference_model_2_layer_call_and_return_conditional_losses_10134
input_5&
sequential_2_10112:
��!
sequential_2_10114:	�&
sequential_2_10116:
��!
sequential_2_10118:	�&
sequential_2_10120:
��!
sequential_2_10122:	�&
sequential_2_10124:
��!
sequential_2_10126:	�%
sequential_2_10128:	�d 
sequential_2_10130:d
identity��$sequential_2/StatefulPartitionedCall�
$sequential_2/StatefulPartitionedCallStatefulPartitionedCallinput_5sequential_2_10112sequential_2_10114sequential_2_10116sequential_2_10118sequential_2_10120sequential_2_10122sequential_2_10124sequential_2_10126sequential_2_10128sequential_2_10130*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������d*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_sequential_2_layer_call_and_return_conditional_losses_9989�
IdentityIdentity-sequential_2/StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������dI
NoOpNoOp%^sequential_2/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:�������������������: : : : : : : : : : 2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall:%
!

_user_specified_name10130:%	!

_user_specified_name10128:%!

_user_specified_name10126:%!

_user_specified_name10124:%!

_user_specified_name10122:%!

_user_specified_name10120:%!

_user_specified_name10118:%!

_user_specified_name10116:%!

_user_specified_name10114:%!

_user_specified_name10112:^ Z
5
_output_shapes#
!:�������������������
!
_user_specified_name	input_5
�
�
'__inference_model_2_layer_call_fn_10184
input_5
unknown:
��
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:	�d
	unknown_8:d
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������d*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_model_2_layer_call_and_return_conditional_losses_10134|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������d<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:�������������������: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%
!

_user_specified_name10180:%	!

_user_specified_name10178:%!

_user_specified_name10176:%!

_user_specified_name10174:%!

_user_specified_name10172:%!

_user_specified_name10170:%!

_user_specified_name10168:%!

_user_specified_name10166:%!

_user_specified_name10164:%!

_user_specified_name10162:^ Z
5
_output_shapes#
!:�������������������
!
_user_specified_name	input_5
�
�
(__inference_dense_20_layer_call_fn_10381

inputs
unknown:	�d
	unknown_0:d
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_20_layer_call_and_return_conditional_losses_9953|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������d<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:�������������������: : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name10377:%!

_user_specified_name10375:] Y
5
_output_shapes#
!:�������������������
 
_user_specified_nameinputs
�
�
B__inference_dense_20_layer_call_and_return_conditional_losses_9953

inputs4
!tensordot_readvariableop_resource:	�d-
biasadd_readvariableop_resource:d
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	�d*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*5
_output_shapes#
!:��������������������
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:dY
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :������������������dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :������������������d]
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :������������������dn
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :������������������dV
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:�������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:] Y
5
_output_shapes#
!:�������������������
 
_user_specified_nameinputs
�
�
C__inference_dense_18_layer_call_and_return_conditional_losses_10332

inputs5
!tensordot_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp|
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*5
_output_shapes#
!:��������������������
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*5
_output_shapes#
!:�������������������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:�������������������^
ReluReluBiasAdd:output:0*
T0*5
_output_shapes#
!:�������������������o
IdentityIdentityRelu:activations:0^NoOp*
T0*5
_output_shapes#
!:�������������������V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:�������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:] Y
5
_output_shapes#
!:�������������������
 
_user_specified_nameinputs
�J
�	
!__inference__traced_restore_10587
file_prefix4
 assignvariableop_dense_16_kernel:
��/
 assignvariableop_1_dense_16_bias:	�6
"assignvariableop_2_dense_17_kernel:
��/
 assignvariableop_3_dense_17_bias:	�6
"assignvariableop_4_dense_18_kernel:
��/
 assignvariableop_5_dense_18_bias:	�6
"assignvariableop_6_dense_19_kernel:
��/
 assignvariableop_7_dense_19_bias:	�5
"assignvariableop_8_dense_20_kernel:	�d.
 assignvariableop_9_dense_20_bias:d'
assignvariableop_10_iteration:	 +
!assignvariableop_11_learning_rate: %
assignvariableop_12_total_1: %
assignvariableop_13_count_1: #
assignvariableop_14_total: #
assignvariableop_15_count: 
identity_17��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*5
value,B*B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*X
_output_shapesF
D:::::::::::::::::*
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp assignvariableop_dense_16_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_16_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_17_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_17_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_18_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_18_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_19_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_19_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp"assignvariableop_8_dense_20_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp assignvariableop_9_dense_20_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_10AssignVariableOpassignvariableop_10_iterationIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp!assignvariableop_11_learning_rateIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOpassignvariableop_12_total_1Identity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOpassignvariableop_13_count_1Identity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOpassignvariableop_14_totalIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOpassignvariableop_15_countIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_16Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_17IdentityIdentity_16:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_17Identity_17:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
": : : : : : : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:%!

_user_specified_namecount:%!

_user_specified_nametotal:'#
!
_user_specified_name	count_1:'#
!
_user_specified_name	total_1:-)
'
_user_specified_namelearning_rate:)%
#
_user_specified_name	iteration:-
)
'
_user_specified_namedense_20/bias:/	+
)
_user_specified_namedense_20/kernel:-)
'
_user_specified_namedense_19/bias:/+
)
_user_specified_namedense_19/kernel:-)
'
_user_specified_namedense_18/bias:/+
)
_user_specified_namedense_18/kernel:-)
'
_user_specified_namedense_17/bias:/+
)
_user_specified_namedense_17/kernel:-)
'
_user_specified_namedense_16/bias:/+
)
_user_specified_namedense_16/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
B__inference_model_2_layer_call_and_return_conditional_losses_10109
input_5&
sequential_2_10087:
��!
sequential_2_10089:	�&
sequential_2_10091:
��!
sequential_2_10093:	�&
sequential_2_10095:
��!
sequential_2_10097:	�&
sequential_2_10099:
��!
sequential_2_10101:	�%
sequential_2_10103:	�d 
sequential_2_10105:d
identity��$sequential_2/StatefulPartitionedCall�
$sequential_2/StatefulPartitionedCallStatefulPartitionedCallinput_5sequential_2_10087sequential_2_10089sequential_2_10091sequential_2_10093sequential_2_10095sequential_2_10097sequential_2_10099sequential_2_10101sequential_2_10103sequential_2_10105*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������d*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_sequential_2_layer_call_and_return_conditional_losses_9960�
IdentityIdentity-sequential_2/StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������dI
NoOpNoOp%^sequential_2/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:�������������������: : : : : : : : : : 2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall:%
!

_user_specified_name10105:%	!

_user_specified_name10103:%!

_user_specified_name10101:%!

_user_specified_name10099:%!

_user_specified_name10097:%!

_user_specified_name10095:%!

_user_specified_name10093:%!

_user_specified_name10091:%!

_user_specified_name10089:%!

_user_specified_name10087:^ Z
5
_output_shapes#
!:�������������������
!
_user_specified_name	input_5
�
�
C__inference_dense_19_layer_call_and_return_conditional_losses_10372

inputs5
!tensordot_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp|
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*5
_output_shapes#
!:��������������������
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*5
_output_shapes#
!:�������������������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:�������������������^
ReluReluBiasAdd:output:0*
T0*5
_output_shapes#
!:�������������������o
IdentityIdentityRelu:activations:0^NoOp*
T0*5
_output_shapes#
!:�������������������V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:�������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:] Y
5
_output_shapes#
!:�������������������
 
_user_specified_nameinputs
�
�
__inference__traced_save_10530
file_prefix:
&read_disablecopyonread_dense_16_kernel:
��5
&read_1_disablecopyonread_dense_16_bias:	�<
(read_2_disablecopyonread_dense_17_kernel:
��5
&read_3_disablecopyonread_dense_17_bias:	�<
(read_4_disablecopyonread_dense_18_kernel:
��5
&read_5_disablecopyonread_dense_18_bias:	�<
(read_6_disablecopyonread_dense_19_kernel:
��5
&read_7_disablecopyonread_dense_19_bias:	�;
(read_8_disablecopyonread_dense_20_kernel:	�d4
&read_9_disablecopyonread_dense_20_bias:d-
#read_10_disablecopyonread_iteration:	 1
'read_11_disablecopyonread_learning_rate: +
!read_12_disablecopyonread_total_1: +
!read_13_disablecopyonread_count_1: )
read_14_disablecopyonread_total: )
read_15_disablecopyonread_count: 
savev2_const
identity_33��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
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
: L

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
: x
Read/DisableCopyOnReadDisableCopyOnRead&read_disablecopyonread_dense_16_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp&read_disablecopyonread_dense_16_kernel^Read/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0k
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��c

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��z
Read_1/DisableCopyOnReadDisableCopyOnRead&read_1_disablecopyonread_dense_16_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp&read_1_disablecopyonread_dense_16_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0j

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�`

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes	
:�|
Read_2/DisableCopyOnReadDisableCopyOnRead(read_2_disablecopyonread_dense_17_kernel"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp(read_2_disablecopyonread_dense_17_kernel^Read_2/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0o

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��e

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��z
Read_3/DisableCopyOnReadDisableCopyOnRead&read_3_disablecopyonread_dense_17_bias"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp&read_3_disablecopyonread_dense_17_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0j

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�`

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes	
:�|
Read_4/DisableCopyOnReadDisableCopyOnRead(read_4_disablecopyonread_dense_18_kernel"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp(read_4_disablecopyonread_dense_18_kernel^Read_4/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0o

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��e

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��z
Read_5/DisableCopyOnReadDisableCopyOnRead&read_5_disablecopyonread_dense_18_bias"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp&read_5_disablecopyonread_dense_18_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0k
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes	
:�|
Read_6/DisableCopyOnReadDisableCopyOnRead(read_6_disablecopyonread_dense_19_kernel"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp(read_6_disablecopyonread_dense_19_kernel^Read_6/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0p
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��z
Read_7/DisableCopyOnReadDisableCopyOnRead&read_7_disablecopyonread_dense_19_bias"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp&read_7_disablecopyonread_dense_19_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0k
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes	
:�|
Read_8/DisableCopyOnReadDisableCopyOnRead(read_8_disablecopyonread_dense_20_kernel"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp(read_8_disablecopyonread_dense_20_kernel^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�d*
dtype0o
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�df
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes
:	�dz
Read_9/DisableCopyOnReadDisableCopyOnRead&read_9_disablecopyonread_dense_20_bias"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp&read_9_disablecopyonread_dense_20_bias^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:d*
dtype0j
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:da
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:dx
Read_10/DisableCopyOnReadDisableCopyOnRead#read_10_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp#read_10_disablecopyonread_iteration^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	g
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0	*
_output_shapes
: |
Read_11/DisableCopyOnReadDisableCopyOnRead'read_11_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp'read_11_disablecopyonread_learning_rate^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_12/DisableCopyOnReadDisableCopyOnRead!read_12_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp!read_12_disablecopyonread_total_1^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_13/DisableCopyOnReadDisableCopyOnRead!read_13_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp!read_13_disablecopyonread_count_1^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_14/DisableCopyOnReadDisableCopyOnReadread_14_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOpread_14_disablecopyonread_total^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_15/DisableCopyOnReadDisableCopyOnReadread_15_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOpread_15_disablecopyonread_count^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*5
value,B*B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtypes
2	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_32Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_33IdentityIdentity_32:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "#
identity_33Identity_33:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$: : : : : : : : : : : : : : : : : : 2(
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
Read_15/ReadVariableOpRead_15/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:=9

_output_shapes
: 

_user_specified_nameConst:%!

_user_specified_namecount:%!

_user_specified_nametotal:'#
!
_user_specified_name	count_1:'#
!
_user_specified_name	total_1:-)
'
_user_specified_namelearning_rate:)%
#
_user_specified_name	iteration:-
)
'
_user_specified_namedense_20/bias:/	+
)
_user_specified_namedense_20/kernel:-)
'
_user_specified_namedense_19/bias:/+
)
_user_specified_namedense_19/kernel:-)
'
_user_specified_namedense_18/bias:/+
)
_user_specified_namedense_18/kernel:-)
'
_user_specified_namedense_17/bias:/+
)
_user_specified_namedense_17/kernel:-)
'
_user_specified_namedense_16/bias:/+
)
_user_specified_namedense_16/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
,__inference_sequential_2_layer_call_fn_10014
dense_16_input
unknown:
��
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:	�d
	unknown_8:d
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_16_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������d*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_sequential_2_layer_call_and_return_conditional_losses_9960|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������d<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:�������������������: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%
!

_user_specified_name10010:%	!

_user_specified_name10008:%!

_user_specified_name10006:%!

_user_specified_name10004:%!

_user_specified_name10002:%!

_user_specified_name10000:$ 

_user_specified_name9998:$ 

_user_specified_name9996:$ 

_user_specified_name9994:$ 

_user_specified_name9992:e a
5
_output_shapes#
!:�������������������
(
_user_specified_namedense_16_input
�
�
F__inference_sequential_2_layer_call_and_return_conditional_losses_9989
dense_16_input!
dense_16_9963:
��
dense_16_9965:	�!
dense_17_9968:
��
dense_17_9970:	�!
dense_18_9973:
��
dense_18_9975:	�!
dense_19_9978:
��
dense_19_9980:	� 
dense_20_9983:	�d
dense_20_9985:d
identity�� dense_16/StatefulPartitionedCall� dense_17/StatefulPartitionedCall� dense_18/StatefulPartitionedCall� dense_19/StatefulPartitionedCall� dense_20/StatefulPartitionedCall�
 dense_16/StatefulPartitionedCallStatefulPartitionedCalldense_16_inputdense_16_9963dense_16_9965*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_16_layer_call_and_return_conditional_losses_9809�
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_9968dense_17_9970*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_17_layer_call_and_return_conditional_losses_9845�
 dense_18/StatefulPartitionedCallStatefulPartitionedCall)dense_17/StatefulPartitionedCall:output:0dense_18_9973dense_18_9975*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_18_layer_call_and_return_conditional_losses_9881�
 dense_19/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0dense_19_9978dense_19_9980*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_19_layer_call_and_return_conditional_losses_9917�
 dense_20/StatefulPartitionedCallStatefulPartitionedCall)dense_19/StatefulPartitionedCall:output:0dense_20_9983dense_20_9985*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_20_layer_call_and_return_conditional_losses_9953�
IdentityIdentity)dense_20/StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������d�
NoOpNoOp!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:�������������������: : : : : : : : : : 2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall:$
 

_user_specified_name9985:$	 

_user_specified_name9983:$ 

_user_specified_name9980:$ 

_user_specified_name9978:$ 

_user_specified_name9975:$ 

_user_specified_name9973:$ 

_user_specified_name9970:$ 

_user_specified_name9968:$ 

_user_specified_name9965:$ 

_user_specified_name9963:e a
5
_output_shapes#
!:�������������������
(
_user_specified_namedense_16_input
�
�
(__inference_dense_18_layer_call_fn_10301

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_18_layer_call_and_return_conditional_losses_9881}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:�������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:�������������������: : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name10297:%!

_user_specified_name10295:] Y
5
_output_shapes#
!:�������������������
 
_user_specified_nameinputs
�
�
(__inference_dense_17_layer_call_fn_10261

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_17_layer_call_and_return_conditional_losses_9845}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:�������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:�������������������: : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name10257:%!

_user_specified_name10255:] Y
5
_output_shapes#
!:�������������������
 
_user_specified_nameinputs
�
�
(__inference_dense_19_layer_call_fn_10341

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_19_layer_call_and_return_conditional_losses_9917}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:�������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:�������������������: : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name10337:%!

_user_specified_name10335:] Y
5
_output_shapes#
!:�������������������
 
_user_specified_nameinputs
�
�
F__inference_sequential_2_layer_call_and_return_conditional_losses_9960
dense_16_input!
dense_16_9810:
��
dense_16_9812:	�!
dense_17_9846:
��
dense_17_9848:	�!
dense_18_9882:
��
dense_18_9884:	�!
dense_19_9918:
��
dense_19_9920:	� 
dense_20_9954:	�d
dense_20_9956:d
identity�� dense_16/StatefulPartitionedCall� dense_17/StatefulPartitionedCall� dense_18/StatefulPartitionedCall� dense_19/StatefulPartitionedCall� dense_20/StatefulPartitionedCall�
 dense_16/StatefulPartitionedCallStatefulPartitionedCalldense_16_inputdense_16_9810dense_16_9812*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_16_layer_call_and_return_conditional_losses_9809�
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_9846dense_17_9848*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_17_layer_call_and_return_conditional_losses_9845�
 dense_18/StatefulPartitionedCallStatefulPartitionedCall)dense_17/StatefulPartitionedCall:output:0dense_18_9882dense_18_9884*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_18_layer_call_and_return_conditional_losses_9881�
 dense_19/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0dense_19_9918dense_19_9920*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_19_layer_call_and_return_conditional_losses_9917�
 dense_20/StatefulPartitionedCallStatefulPartitionedCall)dense_19/StatefulPartitionedCall:output:0dense_20_9954dense_20_9956*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_20_layer_call_and_return_conditional_losses_9953�
IdentityIdentity)dense_20/StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������d�
NoOpNoOp!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:�������������������: : : : : : : : : : 2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall:$
 

_user_specified_name9956:$	 

_user_specified_name9954:$ 

_user_specified_name9920:$ 

_user_specified_name9918:$ 

_user_specified_name9884:$ 

_user_specified_name9882:$ 

_user_specified_name9848:$ 

_user_specified_name9846:$ 

_user_specified_name9812:$ 

_user_specified_name9810:e a
5
_output_shapes#
!:�������������������
(
_user_specified_namedense_16_input
��
�
__inference__wrapped_model_9776
input_5S
?model_2_sequential_2_dense_16_tensordot_readvariableop_resource:
��L
=model_2_sequential_2_dense_16_biasadd_readvariableop_resource:	�S
?model_2_sequential_2_dense_17_tensordot_readvariableop_resource:
��L
=model_2_sequential_2_dense_17_biasadd_readvariableop_resource:	�S
?model_2_sequential_2_dense_18_tensordot_readvariableop_resource:
��L
=model_2_sequential_2_dense_18_biasadd_readvariableop_resource:	�S
?model_2_sequential_2_dense_19_tensordot_readvariableop_resource:
��L
=model_2_sequential_2_dense_19_biasadd_readvariableop_resource:	�R
?model_2_sequential_2_dense_20_tensordot_readvariableop_resource:	�dK
=model_2_sequential_2_dense_20_biasadd_readvariableop_resource:d
identity��4model_2/sequential_2/dense_16/BiasAdd/ReadVariableOp�6model_2/sequential_2/dense_16/Tensordot/ReadVariableOp�4model_2/sequential_2/dense_17/BiasAdd/ReadVariableOp�6model_2/sequential_2/dense_17/Tensordot/ReadVariableOp�4model_2/sequential_2/dense_18/BiasAdd/ReadVariableOp�6model_2/sequential_2/dense_18/Tensordot/ReadVariableOp�4model_2/sequential_2/dense_19/BiasAdd/ReadVariableOp�6model_2/sequential_2/dense_19/Tensordot/ReadVariableOp�4model_2/sequential_2/dense_20/BiasAdd/ReadVariableOp�6model_2/sequential_2/dense_20/Tensordot/ReadVariableOp�
6model_2/sequential_2/dense_16/Tensordot/ReadVariableOpReadVariableOp?model_2_sequential_2_dense_16_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0v
,model_2/sequential_2/dense_16/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:}
,model_2/sequential_2/dense_16/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       r
-model_2/sequential_2/dense_16/Tensordot/ShapeShapeinput_5*
T0*
_output_shapes
::��w
5model_2/sequential_2/dense_16/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
0model_2/sequential_2/dense_16/Tensordot/GatherV2GatherV26model_2/sequential_2/dense_16/Tensordot/Shape:output:05model_2/sequential_2/dense_16/Tensordot/free:output:0>model_2/sequential_2/dense_16/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:y
7model_2/sequential_2/dense_16/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
2model_2/sequential_2/dense_16/Tensordot/GatherV2_1GatherV26model_2/sequential_2/dense_16/Tensordot/Shape:output:05model_2/sequential_2/dense_16/Tensordot/axes:output:0@model_2/sequential_2/dense_16/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:w
-model_2/sequential_2/dense_16/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
,model_2/sequential_2/dense_16/Tensordot/ProdProd9model_2/sequential_2/dense_16/Tensordot/GatherV2:output:06model_2/sequential_2/dense_16/Tensordot/Const:output:0*
T0*
_output_shapes
: y
/model_2/sequential_2/dense_16/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
.model_2/sequential_2/dense_16/Tensordot/Prod_1Prod;model_2/sequential_2/dense_16/Tensordot/GatherV2_1:output:08model_2/sequential_2/dense_16/Tensordot/Const_1:output:0*
T0*
_output_shapes
: u
3model_2/sequential_2/dense_16/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
.model_2/sequential_2/dense_16/Tensordot/concatConcatV25model_2/sequential_2/dense_16/Tensordot/free:output:05model_2/sequential_2/dense_16/Tensordot/axes:output:0<model_2/sequential_2/dense_16/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
-model_2/sequential_2/dense_16/Tensordot/stackPack5model_2/sequential_2/dense_16/Tensordot/Prod:output:07model_2/sequential_2/dense_16/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
1model_2/sequential_2/dense_16/Tensordot/transpose	Transposeinput_57model_2/sequential_2/dense_16/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:��������������������
/model_2/sequential_2/dense_16/Tensordot/ReshapeReshape5model_2/sequential_2/dense_16/Tensordot/transpose:y:06model_2/sequential_2/dense_16/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
.model_2/sequential_2/dense_16/Tensordot/MatMulMatMul8model_2/sequential_2/dense_16/Tensordot/Reshape:output:0>model_2/sequential_2/dense_16/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������z
/model_2/sequential_2/dense_16/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�w
5model_2/sequential_2/dense_16/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
0model_2/sequential_2/dense_16/Tensordot/concat_1ConcatV29model_2/sequential_2/dense_16/Tensordot/GatherV2:output:08model_2/sequential_2/dense_16/Tensordot/Const_2:output:0>model_2/sequential_2/dense_16/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
'model_2/sequential_2/dense_16/TensordotReshape8model_2/sequential_2/dense_16/Tensordot/MatMul:product:09model_2/sequential_2/dense_16/Tensordot/concat_1:output:0*
T0*5
_output_shapes#
!:��������������������
4model_2/sequential_2/dense_16/BiasAdd/ReadVariableOpReadVariableOp=model_2_sequential_2_dense_16_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
%model_2/sequential_2/dense_16/BiasAddBiasAdd0model_2/sequential_2/dense_16/Tensordot:output:0<model_2/sequential_2/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
"model_2/sequential_2/dense_16/ReluRelu.model_2/sequential_2/dense_16/BiasAdd:output:0*
T0*5
_output_shapes#
!:��������������������
6model_2/sequential_2/dense_17/Tensordot/ReadVariableOpReadVariableOp?model_2_sequential_2_dense_17_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0v
,model_2/sequential_2/dense_17/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:}
,model_2/sequential_2/dense_17/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
-model_2/sequential_2/dense_17/Tensordot/ShapeShape0model_2/sequential_2/dense_16/Relu:activations:0*
T0*
_output_shapes
::��w
5model_2/sequential_2/dense_17/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
0model_2/sequential_2/dense_17/Tensordot/GatherV2GatherV26model_2/sequential_2/dense_17/Tensordot/Shape:output:05model_2/sequential_2/dense_17/Tensordot/free:output:0>model_2/sequential_2/dense_17/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:y
7model_2/sequential_2/dense_17/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
2model_2/sequential_2/dense_17/Tensordot/GatherV2_1GatherV26model_2/sequential_2/dense_17/Tensordot/Shape:output:05model_2/sequential_2/dense_17/Tensordot/axes:output:0@model_2/sequential_2/dense_17/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:w
-model_2/sequential_2/dense_17/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
,model_2/sequential_2/dense_17/Tensordot/ProdProd9model_2/sequential_2/dense_17/Tensordot/GatherV2:output:06model_2/sequential_2/dense_17/Tensordot/Const:output:0*
T0*
_output_shapes
: y
/model_2/sequential_2/dense_17/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
.model_2/sequential_2/dense_17/Tensordot/Prod_1Prod;model_2/sequential_2/dense_17/Tensordot/GatherV2_1:output:08model_2/sequential_2/dense_17/Tensordot/Const_1:output:0*
T0*
_output_shapes
: u
3model_2/sequential_2/dense_17/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
.model_2/sequential_2/dense_17/Tensordot/concatConcatV25model_2/sequential_2/dense_17/Tensordot/free:output:05model_2/sequential_2/dense_17/Tensordot/axes:output:0<model_2/sequential_2/dense_17/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
-model_2/sequential_2/dense_17/Tensordot/stackPack5model_2/sequential_2/dense_17/Tensordot/Prod:output:07model_2/sequential_2/dense_17/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
1model_2/sequential_2/dense_17/Tensordot/transpose	Transpose0model_2/sequential_2/dense_16/Relu:activations:07model_2/sequential_2/dense_17/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:��������������������
/model_2/sequential_2/dense_17/Tensordot/ReshapeReshape5model_2/sequential_2/dense_17/Tensordot/transpose:y:06model_2/sequential_2/dense_17/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
.model_2/sequential_2/dense_17/Tensordot/MatMulMatMul8model_2/sequential_2/dense_17/Tensordot/Reshape:output:0>model_2/sequential_2/dense_17/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������z
/model_2/sequential_2/dense_17/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�w
5model_2/sequential_2/dense_17/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
0model_2/sequential_2/dense_17/Tensordot/concat_1ConcatV29model_2/sequential_2/dense_17/Tensordot/GatherV2:output:08model_2/sequential_2/dense_17/Tensordot/Const_2:output:0>model_2/sequential_2/dense_17/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
'model_2/sequential_2/dense_17/TensordotReshape8model_2/sequential_2/dense_17/Tensordot/MatMul:product:09model_2/sequential_2/dense_17/Tensordot/concat_1:output:0*
T0*5
_output_shapes#
!:��������������������
4model_2/sequential_2/dense_17/BiasAdd/ReadVariableOpReadVariableOp=model_2_sequential_2_dense_17_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
%model_2/sequential_2/dense_17/BiasAddBiasAdd0model_2/sequential_2/dense_17/Tensordot:output:0<model_2/sequential_2/dense_17/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
"model_2/sequential_2/dense_17/ReluRelu.model_2/sequential_2/dense_17/BiasAdd:output:0*
T0*5
_output_shapes#
!:��������������������
6model_2/sequential_2/dense_18/Tensordot/ReadVariableOpReadVariableOp?model_2_sequential_2_dense_18_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0v
,model_2/sequential_2/dense_18/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:}
,model_2/sequential_2/dense_18/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
-model_2/sequential_2/dense_18/Tensordot/ShapeShape0model_2/sequential_2/dense_17/Relu:activations:0*
T0*
_output_shapes
::��w
5model_2/sequential_2/dense_18/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
0model_2/sequential_2/dense_18/Tensordot/GatherV2GatherV26model_2/sequential_2/dense_18/Tensordot/Shape:output:05model_2/sequential_2/dense_18/Tensordot/free:output:0>model_2/sequential_2/dense_18/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:y
7model_2/sequential_2/dense_18/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
2model_2/sequential_2/dense_18/Tensordot/GatherV2_1GatherV26model_2/sequential_2/dense_18/Tensordot/Shape:output:05model_2/sequential_2/dense_18/Tensordot/axes:output:0@model_2/sequential_2/dense_18/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:w
-model_2/sequential_2/dense_18/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
,model_2/sequential_2/dense_18/Tensordot/ProdProd9model_2/sequential_2/dense_18/Tensordot/GatherV2:output:06model_2/sequential_2/dense_18/Tensordot/Const:output:0*
T0*
_output_shapes
: y
/model_2/sequential_2/dense_18/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
.model_2/sequential_2/dense_18/Tensordot/Prod_1Prod;model_2/sequential_2/dense_18/Tensordot/GatherV2_1:output:08model_2/sequential_2/dense_18/Tensordot/Const_1:output:0*
T0*
_output_shapes
: u
3model_2/sequential_2/dense_18/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
.model_2/sequential_2/dense_18/Tensordot/concatConcatV25model_2/sequential_2/dense_18/Tensordot/free:output:05model_2/sequential_2/dense_18/Tensordot/axes:output:0<model_2/sequential_2/dense_18/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
-model_2/sequential_2/dense_18/Tensordot/stackPack5model_2/sequential_2/dense_18/Tensordot/Prod:output:07model_2/sequential_2/dense_18/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
1model_2/sequential_2/dense_18/Tensordot/transpose	Transpose0model_2/sequential_2/dense_17/Relu:activations:07model_2/sequential_2/dense_18/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:��������������������
/model_2/sequential_2/dense_18/Tensordot/ReshapeReshape5model_2/sequential_2/dense_18/Tensordot/transpose:y:06model_2/sequential_2/dense_18/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
.model_2/sequential_2/dense_18/Tensordot/MatMulMatMul8model_2/sequential_2/dense_18/Tensordot/Reshape:output:0>model_2/sequential_2/dense_18/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������z
/model_2/sequential_2/dense_18/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�w
5model_2/sequential_2/dense_18/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
0model_2/sequential_2/dense_18/Tensordot/concat_1ConcatV29model_2/sequential_2/dense_18/Tensordot/GatherV2:output:08model_2/sequential_2/dense_18/Tensordot/Const_2:output:0>model_2/sequential_2/dense_18/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
'model_2/sequential_2/dense_18/TensordotReshape8model_2/sequential_2/dense_18/Tensordot/MatMul:product:09model_2/sequential_2/dense_18/Tensordot/concat_1:output:0*
T0*5
_output_shapes#
!:��������������������
4model_2/sequential_2/dense_18/BiasAdd/ReadVariableOpReadVariableOp=model_2_sequential_2_dense_18_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
%model_2/sequential_2/dense_18/BiasAddBiasAdd0model_2/sequential_2/dense_18/Tensordot:output:0<model_2/sequential_2/dense_18/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
"model_2/sequential_2/dense_18/ReluRelu.model_2/sequential_2/dense_18/BiasAdd:output:0*
T0*5
_output_shapes#
!:��������������������
6model_2/sequential_2/dense_19/Tensordot/ReadVariableOpReadVariableOp?model_2_sequential_2_dense_19_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0v
,model_2/sequential_2/dense_19/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:}
,model_2/sequential_2/dense_19/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
-model_2/sequential_2/dense_19/Tensordot/ShapeShape0model_2/sequential_2/dense_18/Relu:activations:0*
T0*
_output_shapes
::��w
5model_2/sequential_2/dense_19/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
0model_2/sequential_2/dense_19/Tensordot/GatherV2GatherV26model_2/sequential_2/dense_19/Tensordot/Shape:output:05model_2/sequential_2/dense_19/Tensordot/free:output:0>model_2/sequential_2/dense_19/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:y
7model_2/sequential_2/dense_19/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
2model_2/sequential_2/dense_19/Tensordot/GatherV2_1GatherV26model_2/sequential_2/dense_19/Tensordot/Shape:output:05model_2/sequential_2/dense_19/Tensordot/axes:output:0@model_2/sequential_2/dense_19/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:w
-model_2/sequential_2/dense_19/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
,model_2/sequential_2/dense_19/Tensordot/ProdProd9model_2/sequential_2/dense_19/Tensordot/GatherV2:output:06model_2/sequential_2/dense_19/Tensordot/Const:output:0*
T0*
_output_shapes
: y
/model_2/sequential_2/dense_19/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
.model_2/sequential_2/dense_19/Tensordot/Prod_1Prod;model_2/sequential_2/dense_19/Tensordot/GatherV2_1:output:08model_2/sequential_2/dense_19/Tensordot/Const_1:output:0*
T0*
_output_shapes
: u
3model_2/sequential_2/dense_19/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
.model_2/sequential_2/dense_19/Tensordot/concatConcatV25model_2/sequential_2/dense_19/Tensordot/free:output:05model_2/sequential_2/dense_19/Tensordot/axes:output:0<model_2/sequential_2/dense_19/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
-model_2/sequential_2/dense_19/Tensordot/stackPack5model_2/sequential_2/dense_19/Tensordot/Prod:output:07model_2/sequential_2/dense_19/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
1model_2/sequential_2/dense_19/Tensordot/transpose	Transpose0model_2/sequential_2/dense_18/Relu:activations:07model_2/sequential_2/dense_19/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:��������������������
/model_2/sequential_2/dense_19/Tensordot/ReshapeReshape5model_2/sequential_2/dense_19/Tensordot/transpose:y:06model_2/sequential_2/dense_19/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
.model_2/sequential_2/dense_19/Tensordot/MatMulMatMul8model_2/sequential_2/dense_19/Tensordot/Reshape:output:0>model_2/sequential_2/dense_19/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������z
/model_2/sequential_2/dense_19/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�w
5model_2/sequential_2/dense_19/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
0model_2/sequential_2/dense_19/Tensordot/concat_1ConcatV29model_2/sequential_2/dense_19/Tensordot/GatherV2:output:08model_2/sequential_2/dense_19/Tensordot/Const_2:output:0>model_2/sequential_2/dense_19/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
'model_2/sequential_2/dense_19/TensordotReshape8model_2/sequential_2/dense_19/Tensordot/MatMul:product:09model_2/sequential_2/dense_19/Tensordot/concat_1:output:0*
T0*5
_output_shapes#
!:��������������������
4model_2/sequential_2/dense_19/BiasAdd/ReadVariableOpReadVariableOp=model_2_sequential_2_dense_19_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
%model_2/sequential_2/dense_19/BiasAddBiasAdd0model_2/sequential_2/dense_19/Tensordot:output:0<model_2/sequential_2/dense_19/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
"model_2/sequential_2/dense_19/ReluRelu.model_2/sequential_2/dense_19/BiasAdd:output:0*
T0*5
_output_shapes#
!:��������������������
6model_2/sequential_2/dense_20/Tensordot/ReadVariableOpReadVariableOp?model_2_sequential_2_dense_20_tensordot_readvariableop_resource*
_output_shapes
:	�d*
dtype0v
,model_2/sequential_2/dense_20/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:}
,model_2/sequential_2/dense_20/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
-model_2/sequential_2/dense_20/Tensordot/ShapeShape0model_2/sequential_2/dense_19/Relu:activations:0*
T0*
_output_shapes
::��w
5model_2/sequential_2/dense_20/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
0model_2/sequential_2/dense_20/Tensordot/GatherV2GatherV26model_2/sequential_2/dense_20/Tensordot/Shape:output:05model_2/sequential_2/dense_20/Tensordot/free:output:0>model_2/sequential_2/dense_20/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:y
7model_2/sequential_2/dense_20/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
2model_2/sequential_2/dense_20/Tensordot/GatherV2_1GatherV26model_2/sequential_2/dense_20/Tensordot/Shape:output:05model_2/sequential_2/dense_20/Tensordot/axes:output:0@model_2/sequential_2/dense_20/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:w
-model_2/sequential_2/dense_20/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
,model_2/sequential_2/dense_20/Tensordot/ProdProd9model_2/sequential_2/dense_20/Tensordot/GatherV2:output:06model_2/sequential_2/dense_20/Tensordot/Const:output:0*
T0*
_output_shapes
: y
/model_2/sequential_2/dense_20/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
.model_2/sequential_2/dense_20/Tensordot/Prod_1Prod;model_2/sequential_2/dense_20/Tensordot/GatherV2_1:output:08model_2/sequential_2/dense_20/Tensordot/Const_1:output:0*
T0*
_output_shapes
: u
3model_2/sequential_2/dense_20/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
.model_2/sequential_2/dense_20/Tensordot/concatConcatV25model_2/sequential_2/dense_20/Tensordot/free:output:05model_2/sequential_2/dense_20/Tensordot/axes:output:0<model_2/sequential_2/dense_20/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
-model_2/sequential_2/dense_20/Tensordot/stackPack5model_2/sequential_2/dense_20/Tensordot/Prod:output:07model_2/sequential_2/dense_20/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
1model_2/sequential_2/dense_20/Tensordot/transpose	Transpose0model_2/sequential_2/dense_19/Relu:activations:07model_2/sequential_2/dense_20/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:��������������������
/model_2/sequential_2/dense_20/Tensordot/ReshapeReshape5model_2/sequential_2/dense_20/Tensordot/transpose:y:06model_2/sequential_2/dense_20/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
.model_2/sequential_2/dense_20/Tensordot/MatMulMatMul8model_2/sequential_2/dense_20/Tensordot/Reshape:output:0>model_2/sequential_2/dense_20/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dy
/model_2/sequential_2/dense_20/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:dw
5model_2/sequential_2/dense_20/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
0model_2/sequential_2/dense_20/Tensordot/concat_1ConcatV29model_2/sequential_2/dense_20/Tensordot/GatherV2:output:08model_2/sequential_2/dense_20/Tensordot/Const_2:output:0>model_2/sequential_2/dense_20/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
'model_2/sequential_2/dense_20/TensordotReshape8model_2/sequential_2/dense_20/Tensordot/MatMul:product:09model_2/sequential_2/dense_20/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :������������������d�
4model_2/sequential_2/dense_20/BiasAdd/ReadVariableOpReadVariableOp=model_2_sequential_2_dense_20_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
%model_2/sequential_2/dense_20/BiasAddBiasAdd0model_2/sequential_2/dense_20/Tensordot:output:0<model_2/sequential_2/dense_20/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :������������������d�
"model_2/sequential_2/dense_20/ReluRelu.model_2/sequential_2/dense_20/BiasAdd:output:0*
T0*4
_output_shapes"
 :������������������d�
IdentityIdentity0model_2/sequential_2/dense_20/Relu:activations:0^NoOp*
T0*4
_output_shapes"
 :������������������d�
NoOpNoOp5^model_2/sequential_2/dense_16/BiasAdd/ReadVariableOp7^model_2/sequential_2/dense_16/Tensordot/ReadVariableOp5^model_2/sequential_2/dense_17/BiasAdd/ReadVariableOp7^model_2/sequential_2/dense_17/Tensordot/ReadVariableOp5^model_2/sequential_2/dense_18/BiasAdd/ReadVariableOp7^model_2/sequential_2/dense_18/Tensordot/ReadVariableOp5^model_2/sequential_2/dense_19/BiasAdd/ReadVariableOp7^model_2/sequential_2/dense_19/Tensordot/ReadVariableOp5^model_2/sequential_2/dense_20/BiasAdd/ReadVariableOp7^model_2/sequential_2/dense_20/Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:�������������������: : : : : : : : : : 2l
4model_2/sequential_2/dense_16/BiasAdd/ReadVariableOp4model_2/sequential_2/dense_16/BiasAdd/ReadVariableOp2p
6model_2/sequential_2/dense_16/Tensordot/ReadVariableOp6model_2/sequential_2/dense_16/Tensordot/ReadVariableOp2l
4model_2/sequential_2/dense_17/BiasAdd/ReadVariableOp4model_2/sequential_2/dense_17/BiasAdd/ReadVariableOp2p
6model_2/sequential_2/dense_17/Tensordot/ReadVariableOp6model_2/sequential_2/dense_17/Tensordot/ReadVariableOp2l
4model_2/sequential_2/dense_18/BiasAdd/ReadVariableOp4model_2/sequential_2/dense_18/BiasAdd/ReadVariableOp2p
6model_2/sequential_2/dense_18/Tensordot/ReadVariableOp6model_2/sequential_2/dense_18/Tensordot/ReadVariableOp2l
4model_2/sequential_2/dense_19/BiasAdd/ReadVariableOp4model_2/sequential_2/dense_19/BiasAdd/ReadVariableOp2p
6model_2/sequential_2/dense_19/Tensordot/ReadVariableOp6model_2/sequential_2/dense_19/Tensordot/ReadVariableOp2l
4model_2/sequential_2/dense_20/BiasAdd/ReadVariableOp4model_2/sequential_2/dense_20/BiasAdd/ReadVariableOp2p
6model_2/sequential_2/dense_20/Tensordot/ReadVariableOp6model_2/sequential_2/dense_20/Tensordot/ReadVariableOp:(
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
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:^ Z
5
_output_shapes#
!:�������������������
!
_user_specified_name	input_5
�
�
C__inference_dense_17_layer_call_and_return_conditional_losses_10292

inputs5
!tensordot_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp|
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*5
_output_shapes#
!:��������������������
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*5
_output_shapes#
!:�������������������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:�������������������^
ReluReluBiasAdd:output:0*
T0*5
_output_shapes#
!:�������������������o
IdentityIdentityRelu:activations:0^NoOp*
T0*5
_output_shapes#
!:�������������������V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:�������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:] Y
5
_output_shapes#
!:�������������������
 
_user_specified_nameinputs
�
�
'__inference_model_2_layer_call_fn_10159
input_5
unknown:
��
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:	�d
	unknown_8:d
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������d*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_model_2_layer_call_and_return_conditional_losses_10109|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������d<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:�������������������: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%
!

_user_specified_name10155:%	!

_user_specified_name10153:%!

_user_specified_name10151:%!

_user_specified_name10149:%!

_user_specified_name10147:%!

_user_specified_name10145:%!

_user_specified_name10143:%!

_user_specified_name10141:%!

_user_specified_name10139:%!

_user_specified_name10137:^ Z
5
_output_shapes#
!:�������������������
!
_user_specified_name	input_5
�
�
B__inference_dense_19_layer_call_and_return_conditional_losses_9917

inputs5
!tensordot_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp|
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*5
_output_shapes#
!:��������������������
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*5
_output_shapes#
!:�������������������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:�������������������^
ReluReluBiasAdd:output:0*
T0*5
_output_shapes#
!:�������������������o
IdentityIdentityRelu:activations:0^NoOp*
T0*5
_output_shapes#
!:�������������������V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:�������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:] Y
5
_output_shapes#
!:�������������������
 
_user_specified_nameinputs
�
�
B__inference_dense_18_layer_call_and_return_conditional_losses_9881

inputs5
!tensordot_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp|
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*5
_output_shapes#
!:��������������������
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*5
_output_shapes#
!:�������������������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:�������������������^
ReluReluBiasAdd:output:0*
T0*5
_output_shapes#
!:�������������������o
IdentityIdentityRelu:activations:0^NoOp*
T0*5
_output_shapes#
!:�������������������V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:�������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:] Y
5
_output_shapes#
!:�������������������
 
_user_specified_nameinputs
�
�
C__inference_dense_16_layer_call_and_return_conditional_losses_10252

inputs5
!tensordot_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp|
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*5
_output_shapes#
!:��������������������
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*5
_output_shapes#
!:�������������������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:�������������������^
ReluReluBiasAdd:output:0*
T0*5
_output_shapes#
!:�������������������o
IdentityIdentityRelu:activations:0^NoOp*
T0*5
_output_shapes#
!:�������������������V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:�������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:] Y
5
_output_shapes#
!:�������������������
 
_user_specified_nameinputs
�
�
B__inference_dense_17_layer_call_and_return_conditional_losses_9845

inputs5
!tensordot_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp|
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*5
_output_shapes#
!:��������������������
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*5
_output_shapes#
!:�������������������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:�������������������^
ReluReluBiasAdd:output:0*
T0*5
_output_shapes#
!:�������������������o
IdentityIdentityRelu:activations:0^NoOp*
T0*5
_output_shapes#
!:�������������������V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:�������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:] Y
5
_output_shapes#
!:�������������������
 
_user_specified_nameinputs
�
�
,__inference_sequential_2_layer_call_fn_10039
dense_16_input
unknown:
��
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:	�d
	unknown_8:d
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_16_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������d*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_sequential_2_layer_call_and_return_conditional_losses_9989|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������d<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:�������������������: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%
!

_user_specified_name10035:%	!

_user_specified_name10033:%!

_user_specified_name10031:%!

_user_specified_name10029:%!

_user_specified_name10027:%!

_user_specified_name10025:%!

_user_specified_name10023:%!

_user_specified_name10021:%!

_user_specified_name10019:%!

_user_specified_name10017:e a
5
_output_shapes#
!:�������������������
(
_user_specified_namedense_16_input"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
I
input_5>
serving_default_input_5:0�������������������M
sequential_2=
StatefulPartitionedCall:0������������������dtensorflow/serving/predict:ך
�
layer-0
layer_with_weights-0
layer-1
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
	_default_save_signature

	optimizer

signatures"
_tf_keras_network
"
_tf_keras_input_layer
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_sequential
f
0
1
2
3
4
5
6
7
8
 9"
trackable_list_wrapper
f
0
1
2
3
4
5
6
7
8
 9"
trackable_list_wrapper
 "
trackable_list_wrapper
�
!non_trainable_variables

"layers
#metrics
$layer_regularization_losses
%layer_metrics
	variables
trainable_variables
regularization_losses
__call__
	_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
&trace_0
'trace_12�
'__inference_model_2_layer_call_fn_10159
'__inference_model_2_layer_call_fn_10184�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z&trace_0z'trace_1
�
(trace_0
)trace_12�
B__inference_model_2_layer_call_and_return_conditional_losses_10109
B__inference_model_2_layer_call_and_return_conditional_losses_10134�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z(trace_0z)trace_1
�B�
__inference__wrapped_model_9776input_5"�
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
 
j
*
_variables
+_iterations
,_learning_rate
-_update_step_xla"
experimentalOptimizer
,
.serving_default"
signature_map
�
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
*@&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses

kernel
 bias"
_tf_keras_layer
f
0
1
2
3
4
5
6
7
8
 9"
trackable_list_wrapper
f
0
1
2
3
4
5
6
7
8
 9"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Mnon_trainable_variables

Nlayers
Ometrics
Player_regularization_losses
Qlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Rtrace_0
Strace_12�
,__inference_sequential_2_layer_call_fn_10014
,__inference_sequential_2_layer_call_fn_10039�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zRtrace_0zStrace_1
�
Ttrace_0
Utrace_12�
F__inference_sequential_2_layer_call_and_return_conditional_losses_9960
F__inference_sequential_2_layer_call_and_return_conditional_losses_9989�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zTtrace_0zUtrace_1
#:!
��2dense_16/kernel
:�2dense_16/bias
#:!
��2dense_17/kernel
:�2dense_17/bias
#:!
��2dense_18/kernel
:�2dense_18/bias
#:!
��2dense_19/kernel
:�2dense_19/bias
": 	�d2dense_20/kernel
:d2dense_20/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_model_2_layer_call_fn_10159input_5"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
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
 
�B�
'__inference_model_2_layer_call_fn_10184input_5"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
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
 
�B�
B__inference_model_2_layer_call_and_return_conditional_losses_10109input_5"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
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
 
�B�
B__inference_model_2_layer_call_and_return_conditional_losses_10134input_5"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
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
 
'
+0"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
�2��
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
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
 0
�B�
#__inference_signature_wrapper_10212input_5"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs�
	jinput_5
kwonlydefaults
 
annotations� *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Xnon_trainable_variables

Ylayers
Zmetrics
[layer_regularization_losses
\layer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses"
_generic_user_object
�
]trace_02�
(__inference_dense_16_layer_call_fn_10221�
���
FullArgSpec
args�

jinputs
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
 z]trace_0
�
^trace_02�
C__inference_dense_16_layer_call_and_return_conditional_losses_10252�
���
FullArgSpec
args�

jinputs
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
 z^trace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses"
_generic_user_object
�
dtrace_02�
(__inference_dense_17_layer_call_fn_10261�
���
FullArgSpec
args�

jinputs
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
 zdtrace_0
�
etrace_02�
C__inference_dense_17_layer_call_and_return_conditional_losses_10292�
���
FullArgSpec
args�

jinputs
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
 zetrace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
fnon_trainable_variables

glayers
hmetrics
ilayer_regularization_losses
jlayer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses"
_generic_user_object
�
ktrace_02�
(__inference_dense_18_layer_call_fn_10301�
���
FullArgSpec
args�

jinputs
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
 zktrace_0
�
ltrace_02�
C__inference_dense_18_layer_call_and_return_conditional_losses_10332�
���
FullArgSpec
args�

jinputs
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
 zltrace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses"
_generic_user_object
�
rtrace_02�
(__inference_dense_19_layer_call_fn_10341�
���
FullArgSpec
args�

jinputs
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
 zrtrace_0
�
strace_02�
C__inference_dense_19_layer_call_and_return_conditional_losses_10372�
���
FullArgSpec
args�

jinputs
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
 zstrace_0
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
tnon_trainable_variables

ulayers
vmetrics
wlayer_regularization_losses
xlayer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
�
ytrace_02�
(__inference_dense_20_layer_call_fn_10381�
���
FullArgSpec
args�

jinputs
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
 zytrace_0
�
ztrace_02�
C__inference_dense_20_layer_call_and_return_conditional_losses_10412�
���
FullArgSpec
args�

jinputs
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
 zztrace_0
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_sequential_2_layer_call_fn_10014dense_16_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
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
 
�B�
,__inference_sequential_2_layer_call_fn_10039dense_16_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
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
 
�B�
F__inference_sequential_2_layer_call_and_return_conditional_losses_9960dense_16_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
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
 
�B�
F__inference_sequential_2_layer_call_and_return_conditional_losses_9989dense_16_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
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
 
N
{	variables
|	keras_api
	}total
	~count"
_tf_keras_metric
b
	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense_16_layer_call_fn_10221inputs"�
���
FullArgSpec
args�

jinputs
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
 
�B�
C__inference_dense_16_layer_call_and_return_conditional_losses_10252inputs"�
���
FullArgSpec
args�

jinputs
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
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense_17_layer_call_fn_10261inputs"�
���
FullArgSpec
args�

jinputs
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
 
�B�
C__inference_dense_17_layer_call_and_return_conditional_losses_10292inputs"�
���
FullArgSpec
args�

jinputs
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
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense_18_layer_call_fn_10301inputs"�
���
FullArgSpec
args�

jinputs
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
 
�B�
C__inference_dense_18_layer_call_and_return_conditional_losses_10332inputs"�
���
FullArgSpec
args�

jinputs
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
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense_19_layer_call_fn_10341inputs"�
���
FullArgSpec
args�

jinputs
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
 
�B�
C__inference_dense_19_layer_call_and_return_conditional_losses_10372inputs"�
���
FullArgSpec
args�

jinputs
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
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense_20_layer_call_fn_10381inputs"�
���
FullArgSpec
args�

jinputs
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
 
�B�
C__inference_dense_20_layer_call_and_return_conditional_losses_10412inputs"�
���
FullArgSpec
args�

jinputs
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
 
.
}0
~1"
trackable_list_wrapper
-
{	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
-
	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper�
__inference__wrapped_model_9776�
 >�;
4�1
/�,
input_5�������������������
� "H�E
C
sequential_23�0
sequential_2������������������d�
C__inference_dense_16_layer_call_and_return_conditional_losses_10252=�:
3�0
.�+
inputs�������������������
� ":�7
0�-
tensor_0�������������������
� �
(__inference_dense_16_layer_call_fn_10221t=�:
3�0
.�+
inputs�������������������
� "/�,
unknown��������������������
C__inference_dense_17_layer_call_and_return_conditional_losses_10292=�:
3�0
.�+
inputs�������������������
� ":�7
0�-
tensor_0�������������������
� �
(__inference_dense_17_layer_call_fn_10261t=�:
3�0
.�+
inputs�������������������
� "/�,
unknown��������������������
C__inference_dense_18_layer_call_and_return_conditional_losses_10332=�:
3�0
.�+
inputs�������������������
� ":�7
0�-
tensor_0�������������������
� �
(__inference_dense_18_layer_call_fn_10301t=�:
3�0
.�+
inputs�������������������
� "/�,
unknown��������������������
C__inference_dense_19_layer_call_and_return_conditional_losses_10372=�:
3�0
.�+
inputs�������������������
� ":�7
0�-
tensor_0�������������������
� �
(__inference_dense_19_layer_call_fn_10341t=�:
3�0
.�+
inputs�������������������
� "/�,
unknown��������������������
C__inference_dense_20_layer_call_and_return_conditional_losses_10412~ =�:
3�0
.�+
inputs�������������������
� "9�6
/�,
tensor_0������������������d
� �
(__inference_dense_20_layer_call_fn_10381s =�:
3�0
.�+
inputs�������������������
� ".�+
unknown������������������d�
B__inference_model_2_layer_call_and_return_conditional_losses_10109�
 F�C
<�9
/�,
input_5�������������������
p

 
� "9�6
/�,
tensor_0������������������d
� �
B__inference_model_2_layer_call_and_return_conditional_losses_10134�
 F�C
<�9
/�,
input_5�������������������
p 

 
� "9�6
/�,
tensor_0������������������d
� �
'__inference_model_2_layer_call_fn_10159�
 F�C
<�9
/�,
input_5�������������������
p

 
� ".�+
unknown������������������d�
'__inference_model_2_layer_call_fn_10184�
 F�C
<�9
/�,
input_5�������������������
p 

 
� ".�+
unknown������������������d�
F__inference_sequential_2_layer_call_and_return_conditional_losses_9960�
 M�J
C�@
6�3
dense_16_input�������������������
p

 
� "9�6
/�,
tensor_0������������������d
� �
F__inference_sequential_2_layer_call_and_return_conditional_losses_9989�
 M�J
C�@
6�3
dense_16_input�������������������
p 

 
� "9�6
/�,
tensor_0������������������d
� �
,__inference_sequential_2_layer_call_fn_10014�
 M�J
C�@
6�3
dense_16_input�������������������
p

 
� ".�+
unknown������������������d�
,__inference_sequential_2_layer_call_fn_10039�
 M�J
C�@
6�3
dense_16_input�������������������
p 

 
� ".�+
unknown������������������d�
#__inference_signature_wrapper_10212�
 I�F
� 
?�<
:
input_5/�,
input_5�������������������"H�E
C
sequential_23�0
sequential_2������������������d