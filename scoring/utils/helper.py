#creates identity matrix
import torch



def identity(batch_size,dim=3,dtype=torch.float32, device='cpu'):
    """
    Creates an identity matrix of shape (batch_size, dim, dim).
    Remember for 2d Points the dim is 3, as affine transformation require the etra dimension for translation
    If batch size is list tuple or torch Size, it will be unpacked
    """
    #if batch size is a unpackable type, unpack it
    id = torch.eye(dim, dtype=dtype, device=device)
    if isinstance(batch_size, (list, tuple, torch.Size)):
        #check for empty batch size
        if len(batch_size) == 0:
            return id
        id_matrix = id.unsqueeze(0).repeat(*batch_size, 1, 1)
    else:
        id_matrix = id.unsqueeze(0).repeat(batch_size, 1, 1)
    return id_matrix


def get_orbit_member_function(orbit_function):
    def orbit_member_function(n,n_samples=16, domain=None) :
        res =  orbit_function(n_samples=n_samples, domain=domain, extend=0, shift=0)
        return res[n]
    return orbit_member_function







def reflection_matrix(param):
    """
    TODO think about how to limit this to only have a specific number of cases
    :param param:
    :return:
    """
    dim = param.shape[-1]
    batch_size = param.shape[:-1]
    # Create a reflection matrix
    reflection_matrix = identity(batch_size, dim+1, dtype=param.dtype, device=param.device)
    # Fill the reflection matrix

    reflection_matrix[..., :-1, :-1] = torch.diag_embed(param)
    return reflection_matrix


def apply_wrapper(matrix_func):
    def apply_func(T,param):
        if T is None:
            return matrix_func(param)
        else:
            return torch.matmul(matrix_func(param), T)
    return apply_func






