import torch
import torch.func
def estimate_gradient(module,x0,displacement):
    """Estimates the directional derivative of a module's output.

    This function computes the Jacobian-vector product (JVP) of the `module` at
    a given point `x0` in the direction of `displacement`. The JVP is equivalent
    to the directional derivative of the function represented by the module.

    Args:
        module (torch.nn.Module or callable): The model or function to differentiate.
            It should accept a tensor `x0` and return a tensor.
        x0 (torch.Tensor): The point at which to compute the JVP. This is the
            primal input to the module.
        displacement (torch.Tensor): The direction vector for the directional
            derivative. This is the tangent vector `v` in `Jv`.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The output of `module(x0)`.
            - The Jacobian-vector product.
    """
    # Get the parameters and buffers of the module for functional_call
    params = dict(module.named_parameters())
    all_buffers = dict(module.named_buffers())

    # jvp requires primals to be floating-point. Filter out non-float buffers.
    # The functional_call inside the lambda will still see all_buffers from the closure.
    float_buffers = {name: b for name, b in all_buffers.items() if b.is_floating_point()}

    # Define a functional forward pass that takes params, buffers, and input
    # This function will be passed to torch.func.jvp
    def functional_forward_for_jvp(params, float_buffers, x_input):
        # functional_call returns (output, updated_buffers)
        # We only need the output for the JVP calculation.
        output = torch.func.functional_call(module, (params, all_buffers), x_input)
        return output

    # Create zero tangents for parameters and buffers as we are only differentiating w.r.t. x_input
    tangent_params = {name: torch.zeros_like(p) for name, p in params.items()} # No tangent for params
    tangent_buffers = {name: torch.zeros_like(b) for name, b in float_buffers.items()} # No tangent for buffers

    # Call jvp with the functional forward pass
    primal_output, jvp_output = torch.func.jvp(functional_forward_for_jvp, (params, float_buffers, x0), (tangent_params, tangent_buffers, displacement))
    return primal_output, jvp_output