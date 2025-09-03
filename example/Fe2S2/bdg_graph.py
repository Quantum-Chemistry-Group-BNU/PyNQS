# Get the BDG from Exchange Integral Kij
import torch
from libs.C_extension import decompress_h1e_h2e
x = torch.load("./molecule/fe2s2-OO.pth", map_location="cpu", weights_only=False)
h1e, h2e = x["h1e"], x["h2e"]
h1e, h2e = decompress_h1e_h2e(h1e, h2e, 40)
norb = 20
kij_OO = np.zeros((norb,norb))
for i in range(norb):
    for j in range(norb):
        p, q = 2 * i, 2 * i + 1
        r, s = 2 * j, 2 * j + 1
        kij_OO[i, j] = h2e[p, q, r, s]

import networkx as nx
from utils.graph import fielder, nxutils
forder = fielder.orbitalOrdering(kij_OO,mode='kmat',debug=False)
fgraph = nxutils.fromOrderToDiGraph(forder)
# nx.write_graphml_xml(fgraph, "./graph/Fe2S2-OO-maxdes-0.graphml")
maxdes = 1
fgraph1 = nxutils.addEdgesByGreedySearch(fgraph,kij_OO,maxdes)
# nx.write_graphml_xml(fgraph1, "./graph/Fe2S2-OO-maxdes-1.graphml")
