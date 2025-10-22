import torch
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
    fn = lambda x_: module(x_)
    return torch.func.jvp(fn,x0,displacement)