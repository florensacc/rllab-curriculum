require 'securerandom'
require 'fileutils'
require 'time'
require 'shellwords'

def to_param_val(v)
  if v.nil?
    ""
  elsif v == false
    "False"
  elsif v == true
    "True"
  elsif v.respond_to?(:join)
    v.map(&:to_s).map(&:shellescape).join(" ")
  else
    v.to_s.shellescape
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

def to_docker_command(command)
  "/home/ubuntu/dockerfiles/docker_run.sh #{command}"
end

def create_task_script(command, options={})
  launch = options.delete(:launch) || false
  fname = options.delete(:fname)
  prefix = options.delete(:prefix)
  unless fname
    folder = "#{File.dirname(__FILE__)}/../../launch_scripts"
    FileUtils.mkdir_p(folder)
    file_name = "#{SecureRandom.uuid}.sh"
    if prefix
      file_name = prefix + "_" + Time.now.utc.iso8601 + "_" + file_name
    end
    fname = "#{folder}/#{file_name}"
  end
  puts fname
  f = File.open(fname, "w")
  f.puts command
  f.close
  system("chmod +x " + fname)
  if launch
    system("qsub -V -b n -l mem_free=8G,h_vmem=14G -r y -cwd " + fname)
  end
end
