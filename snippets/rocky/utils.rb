def to_param_val(v)
  if v.nil?
    ""
  elsif v == false
    "False"
  elsif v == true
    "True"
  elsif v.respond_to?(:join)
    v.join(" ")
  else
    v
  end
end

def to_command(params)
  command = "python scripts/run_experiment.py"
  params.each do |k, v|
    if v.is_a?(Hash)
      v.each do |nk, nv|
        if nk.to_s == "_name"
          command += " \\\n" + "  --#{k} #{to_param_val(nv)}"
        else
          command += " \\\n" + "  --#{k}_#{nk} #{to_param_val(nv)}"
        end
      end
    else
      command += " \\\n" + "  --#{k} #{to_param_val(v)}"
    end
  end
  command
end
