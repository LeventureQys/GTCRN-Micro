# quantization aware training
from torch.ao.quantization import get_default_qat_qconfig
from torch.ao.quantization.quantize_fx import prepare_qat_fx

from gtcrn_micro.models.gtcrn_micro import GTCRNMicro


def main():
    # floating point model
    model_fp = GTCRNMicro().train()

    # setting up QAT configuration
    qconfig = get_default_qat_qconfig("fbgemm")
    qconfig_dict = {"": qconfig}

    # wrapping the model with fake quant layers
    model_qat = prepare_qat_fx(model_fp, qconfig_dict)
    model_qat.train()
