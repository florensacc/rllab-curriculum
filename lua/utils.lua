function updateTargetParam(param, targetParam, tau)
  targetParam:mul(1.0 - tau):add(tau, param)
end

function updateTarget(module, targetModule, tau)
  if torch.isTypeOf(module, 'nn.Container') then
    for i = 1, #module.modules do
      updateTarget(module.modules[i], targetModule.modules[i], tau)
    end
  else
    if module.weight then
      updateTargetParam(module.weight, targetModule.weight, tau)
      updateTargetParam(module.bias, targetModule.bias, tau)
    end
    if module.running_mean then
      updateTargetParam(module.running_mean, targetModule.running_mean, tau)
      updateTargetParam(module.running_std, targetModule.running_std, tau)
    end
  end
end
