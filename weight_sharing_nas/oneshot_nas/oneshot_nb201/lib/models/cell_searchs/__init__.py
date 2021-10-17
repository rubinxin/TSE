##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
# The macro structure is defined in NAS-Bench-201

from .generic_model         import GenericNAS201Model
from .genotypes             import Structure as CellStructure, architectures as CellArchitectures
# NASNet-based macro structure



nas201_super_nets = {
                     "generic": GenericNAS201Model}

