# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: message.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='message.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n\rmessage.proto\"^\n\x0eImageDataBlock\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x12\n\ndata_block\x18\x03 \x01(\x0c\x12\x17\n\x0f\x64\x61ta_block_hash\x18\x04 \x01(\t\x12\x11\n\tdata_hash\x18\x05 \x01(\t\"4\n\x0c\x44\x65tectResult\x12\x0b\n\x03row\x18\x01 \x01(\x05\x12\x0b\n\x03\x63ol\x18\x02 \x01(\x05\x12\n\n\x02id\x18\x03 \x01(\x05\x32@\n\rImageDetector\x12/\n\x0b\x44\x65tectImage\x12\x0f.ImageDataBlock\x1a\r.DetectResult(\x01\x62\x06proto3')
)




_IMAGEDATABLOCK = _descriptor.Descriptor(
  name='ImageDataBlock',
  full_name='ImageDataBlock',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='ImageDataBlock.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='data_block', full_name='ImageDataBlock.data_block', index=1,
      number=3, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='data_block_hash', full_name='ImageDataBlock.data_block_hash', index=2,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='data_hash', full_name='ImageDataBlock.data_hash', index=3,
      number=5, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=17,
  serialized_end=111,
)


_DETECTRESULT = _descriptor.Descriptor(
  name='DetectResult',
  full_name='DetectResult',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='row', full_name='DetectResult.row', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='col', full_name='DetectResult.col', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='id', full_name='DetectResult.id', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=113,
  serialized_end=165,
)

DESCRIPTOR.message_types_by_name['ImageDataBlock'] = _IMAGEDATABLOCK
DESCRIPTOR.message_types_by_name['DetectResult'] = _DETECTRESULT
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

ImageDataBlock = _reflection.GeneratedProtocolMessageType('ImageDataBlock', (_message.Message,), {
  'DESCRIPTOR' : _IMAGEDATABLOCK,
  '__module__' : 'message_pb2'
  # @@protoc_insertion_point(class_scope:ImageDataBlock)
  })
_sym_db.RegisterMessage(ImageDataBlock)

DetectResult = _reflection.GeneratedProtocolMessageType('DetectResult', (_message.Message,), {
  'DESCRIPTOR' : _DETECTRESULT,
  '__module__' : 'message_pb2'
  # @@protoc_insertion_point(class_scope:DetectResult)
  })
_sym_db.RegisterMessage(DetectResult)



_IMAGEDETECTOR = _descriptor.ServiceDescriptor(
  name='ImageDetector',
  full_name='ImageDetector',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  serialized_start=167,
  serialized_end=231,
  methods=[
  _descriptor.MethodDescriptor(
    name='DetectImage',
    full_name='ImageDetector.DetectImage',
    index=0,
    containing_service=None,
    input_type=_IMAGEDATABLOCK,
    output_type=_DETECTRESULT,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_IMAGEDETECTOR)

DESCRIPTOR.services_by_name['ImageDetector'] = _IMAGEDETECTOR

# @@protoc_insertion_point(module_scope)