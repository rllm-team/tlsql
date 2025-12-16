"""BRIDGE Model Construction
"""

from rllm.nn.conv.graph_conv import GCNConv
from rllm.nn.conv.table_conv import TabTransformerConv
from rllm.nn.models import BRIDGE, TableEncoder, GraphEncoder


def build_bridge_model(num_classes, metadata, emb_size):
    """Build BRIDGE model
    Args:
        num_classes: Number of output classes
        metadata: Table metadata for TabTransformer
        emb_size: Embedding size
    Returns:
        BRIDGE model instance
    """
    t_encoder = TableEncoder(
        in_dim=emb_size,
        out_dim=emb_size,
        table_conv=TabTransformerConv,
        metadata=metadata,
    )
    g_encoder = GraphEncoder(
        in_dim=emb_size,
        out_dim=num_classes,
        graph_conv=GCNConv,
    )
    model = BRIDGE(
        table_encoder=t_encoder,
        graph_encoder=g_encoder,
    )
    return model
