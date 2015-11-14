require 'pry'
require 'active_support/all'

CTYPES_MAP = {
  'int' => 'c_int',
  'mjContact' => 'c_void_p',
  'double' => 'c_double',
  'float' => 'c_float',
  'char' => 'c_char',
  'unsigned char' => 'c_ubyte',
  'unsigned int' => 'c_uint',
}

CTYPES_PTR_MAP = {
  'void' => 'c_void_p',
  'char' => 'c_char_p',
}

DEDEF_MAP = {
  'mjtNum' => 'double',
  'mjtByte' => 'unsigned char',
  'mjNREF' => '2',
  'mjNDYN' => '10',
  'mjNGAIN' => '5',
  'mjNBIAS' => '3',
  'mjNIMP' => '3',
  'mjNEQDATA' => '7',
  'mjNTRN' => '1',
}

NP_DTYPE_MAP = {
  'double' => 'double',
  'float' => 'float',
  'int' => 'int',
  'unsigned char' => 'uint8',
}

RESERVED = %w[global map buffer]

def dereserve(name)
  if RESERVED.include? name
    "#{name}_"
  else
    name
  end
end

def dedef(type)
  DEDEF_MAP[type] || type
end

class String
  def blank_or_comment?
    self.strip.size == 0 || self.strip =~ /^\/\//
  end
end

def struct_regex(name)
  /struct #{name}(\s+[^\n]*)?\n\{(.*?)\};/ms
end

def anon_struct_regex
  /struct(.*?)\{(.*?)\}(.*?);/ms
end

def parse_struct(source, name)
  source =~ struct_regex(name)
  content = $2
  subs = []
  # note that this won't work in general; luckily for us, the _mjVisual struct
  # only has anonymous struct fields and nothing else
  subprops = []
  content.scan(anon_struct_regex) {
    subcontent = $2
    subname = $3
    subs << {
      props: subcontent.lines.map(&:strip).reject(&:blank_or_comment?).map{|x| parse_struct_line(source, x)},
      name: "ANON_#{subname.strip.gsub(/^_/,'').upcase}",
      source: source
    }
    subprops << {
      kind: :anonstruct, 
      type: "ANON_#{subname.strip.gsub(/^_/,'').upcase}",
      name: dereserve(subname.strip),
    }
  }
  rest = content.gsub(anon_struct_regex, '')
  rest = rest.lines.map(&:strip).reject(&:blank_or_comment?)
  parsed = rest.map {|x| parse_struct_line(source, x)}
  {
    props: subprops + parsed,
    name: dereserve(name),
    source: source,
    subs: subs,
  }
end

def parse_struct_line(source, line)
  if line =~ /^(\w+) (\w+) (\w+);/
    {
      kind: :value,
      type: dedef($1 + " " + $2),
      name: $3
    }
  elsif line =~ /^(\w+) (\w+);/
    {
      kind: :value,
      type: dedef($1),
      name: $2
    }
  elsif line =~ /^(\w+?)\* (\w+?);/
    ret = {
      kind: :pointer,
      type: dedef($1),
      name: $2
    }
    # special case
    if ret[:name] == "buffer" && ret[:type] == "void"
      ret[:type] = "unsigned char"
    end
    if line =~ /\/\/.*\((\w+) (\w+)\)$/ # size hint
      $stderr.puts line
      ret[:hint] = [dedef($1)]
    elsif line =~ /\/\/.*\(([\w\*]+) x (\w+)\)$/ # size hint
      ret[:hint] = [dedef($1), dedef($2)]
    elsif line =~ /\/\/.*\((\w+)\)$/ # size hint
      ret[:hint] = [dedef($1)]
    end
    ret
  elsif line =~ /(\w+) (\w+)\[\s*(\w+)\s*\];/
    ret = {
      kind: :array,
      type: dedef($1),
      name: $2,
    }
    size = $3
    if size !~ /\d+/
      size = resolve_id_value(source, size)
    end
    ret[:size] = size
    ret
  else
    binding.pry
  end
end

def resolve_id_value(source, id)
  source =~ /enum .*\{(.*?) #{id} (.*?)\}/ms
  if $1.nil?
    binding.pry
  end
  $1.lines.reject(&:blank_or_comment?).size
end

def to_ctypes_type(prop)
  case prop[:kind]
  when :pointer
    CTYPES_PTR_MAP[prop[:type]] || \
      "POINTER(#{to_ctypes_type(prop.merge(kind: :value))})"
  when :anonstruct
    prop[:type]
  when :array
    "#{to_ctypes_type(prop.merge(kind: :value))} * #{prop[:size]}"
  when :value
    CTYPES_MAP[prop[:type]] || prop[:type].upcase
  else
    binding.pry
    raise :wtf
  end
end


def gen_ctypes_src(struct)
%Q{
class #{struct[:name].gsub(/^_/,'').upcase}(Structure):
    #{(struct[:subs] || []).map{|subs| gen_ctypes_src(subs).split("\n").join("\n    ")}.join("\n    ")}
    _fields_ = [
        #{struct[:props].map{|prop|
          "(\"#{prop[:name]}\", #{to_ctypes_type(prop)}),"
        }.join("\n        ")}
    ]
}
end



def gen_wrapper_src(struct)

  def to_size_factor(struct, hint_elem)
    if hint_elem != hint_elem.downcase && hint_elem.size > 3
      binding.pry
    end
    if hint_elem =~ /\*/
      hint_elem.split("*").map{|x| to_size_factor(struct, x)}.join("*")
    elsif hint_elem =~ /^\d+$/
      hint_elem
    else
      if struct[:props].any?{|x| x[:name] == hint_elem}
        "self.#{hint_elem}"
      else
        "self._size_src.#{hint_elem}"
      end
    end
  end

%Q{
class #{struct[:name].gsub(/^_/,'').camelize}Wrapper(object):
    
    def __init__(self, wrapped, size_src=None):
        self._wrapped = wrapped
        self._size_src = size_src

    @property
    def ptr(self):
        return self._wrapped

    @property
    def obj(self):
        return self._wrapped.contents

    #{struct[:props].map{|prop|
      if prop[:kind] == :array && NP_DTYPE_MAP.include?(prop[:type]) || prop[:kind] == :pointer
        dtype =
          if NP_DTYPE_MAP.include? prop[:type]
            NP_DTYPE_MAP[prop[:type]]
          elsif prop[:name] == "buffer"
            "uint8"
          else
            nil
          end


        if prop[:kind] == :pointer && dtype && prop[:hint]
          count = prop[:hint].map{|hint_elem| to_size_factor(struct, hint_elem)}.join("*")
          shape = "(" + prop[:hint].map{|hint_elem| to_size_factor(struct, hint_elem)}.join(", ") + ", )"
          ctype_type = to_ctypes_type(prop.merge(kind: :value))
        elsif prop[:kind] == :array
          count = prop[:size]
          shape = "(#{count}, )"
          ctype_type = to_ctypes_type(prop.merge(kind: :value))
        else
          $stderr.puts "ignoring #{prop}"
          next
        end
%Q{
@property
def #{prop[:name]}(self):
    arr = np.reshape(np.fromiter(self._wrapped.contents.#{prop[:name]}, dtype=np.#{dtype}, count=(#{count})), #{shape})
    arr.setflags(write=False)
    return arr

@#{prop[:name]}.setter
def #{prop[:name]}(self, value):
    val_ptr = np.array(value).ctypes.data_as(POINTER(#{ctype_type}))
    memmove(self._wrapped.contents.#{prop[:name]}, val_ptr, #{count} * sizeof(#{ctype_type}))
}

      elsif prop[:kind] == :value || prop[:kind] == :anonstruct || prop[:kind] == :array
%Q{
@property
def #{prop[:name]}(self):
    return self._wrapped.contents.#{prop[:name]}

@#{prop[:name]}.setter
def #{prop[:name]}(self, value):
    self._wrapped.contents.#{prop[:name]} = value
}
      else
        binding.pry
      end
    }.compact.map{|propsrc| propsrc.split("\n").join("\n    ")}.join("\n    ")}
}
end

source = open(ARGV.first, 'r').read

source = source.gsub(/\r/, '')
source = source.gsub(/\/\*.*?\*\//, '')

puts %Q{
# AUTO GENERATED. DO NOT CHANGE!
from ctypes import *
import numpy as np
}

structs = %w[_mjrRect _mjvCameraPose _mjrOption _mjrContext _mjvCamera _mjvOption _mjvGeom _mjvLight _mjvObjects _mjOption _mjVisual _mjStatistic _mjData _mjModel].map{|x| parse_struct(source, x) }

structs.each {|s| puts gen_ctypes_src(s) }
structs.each {|s| puts gen_wrapper_src(s) }
