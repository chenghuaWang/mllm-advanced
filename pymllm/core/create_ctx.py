from .._C import create_x86_backend, get_engine_ctx, DeviceTypes


def init_x86_ctx():
    x86_bk = create_x86_backend()
    ctx = get_engine_ctx()
    ctx.register_backend(x86_bk)
    ctx.mem().init_buddy_ctx(DeviceTypes.CPU)
    ctx.mem().init_oc(DeviceTypes.CPU)
