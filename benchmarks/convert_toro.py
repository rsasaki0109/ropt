"""Explicit TORO matrix-order conversion of pinned GTSAM w10000.graph prefixes.

This is a derived between-factor subgraph, not the complete original SLAM task.
EQUIV records require explicit --between-only selection; they are counted in
provenance. Retained poses and relative measurements are unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path

SOURCE = "https://raw.githubusercontent.com/borglab/gtsam/4.2.1/examples/Data/w10000.graph"
SHA256 = "1e88f220bd580a4c2b9fc065358608a53b26b1c30033405990b4cdc83236fc89"


def information_to_g2o(values):
    if len(values) != 6:
        raise ValueError("TORO information needs six entries")
    # TORO: xx,xy,yy,aa,xa,ya; g2o: xx,xy,xa,yy,ya,aa.
    return [values[k] for k in (0, 1, 4, 2, 5, 3)]


def convert(source, output, count, between_only=False):
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SHA256:
        raise ValueError("w10000.graph checksum differs")
    if not 2 <= count <= 10000:
        raise ValueError("prefix size must lie in 2..10000")
    vertices, edges, equivalences = {}, [], []
    for line in raw.decode("utf-8").splitlines():
        fields = line.split("#", 1)[0].split()
        if not fields:
            continue
        if fields[0] == "VERTEX2" and len(fields) == 5:
            key = int(fields[1])
            if key in vertices:
                raise ValueError("duplicate vertex")
            vertices[key] = fields
        elif fields[0] == "EDGE2" and len(fields) == 12:
            edges.append(fields)
        elif fields[0] == "EQUIV" and len(fields) == 3:
            if not between_only:
                raise ValueError("EQUIV records require explicit --between-only subgraph selection")
            equivalences.append((int(fields[1]), int(fields[2])))
        else:
            raise ValueError("unexpected TORO record")
    if sorted(vertices) != list(range(10000)):
        raise ValueError("unexpected source IDs")
    selected = [edge for edge in edges if int(edge[1]) < count and int(edge[2]) < count]
    lines = ["VERTEX_SE2 " + " ".join(vertices[k][1:]) for k in range(count)]
    for edge in selected:
        # TORO: Ixx,Ixy,Iyy,Itheta_theta,Ix_theta,Iy_theta.
        # g2o: Ixx,Ixy,Ix_theta,Iyy,Iy_theta,Itheta_theta.
        info = information_to_g2o(edge[6:])
        lines.append("EDGE_SE2 " + " ".join(edge[1:6] + info))
    encoded = ("\n".join(lines) + "\n").encode("utf-8")
    metadata = {"source": SOURCE, "source_sha256": SHA256, "prefix_vertices": count,
                "retained_edges": len(selected), "output_sha256": hashlib.sha256(encoded).hexdigest(),
                "source_equivalence_records": len(equivalences),
                "omitted_equivalences_within_prefix": sum(a < count and b < count for a, b in equivalences),
                "complete_original_problem": False,
                "transformation": "explicit between-only subgraph: IDs 0..N-1, keep EDGE2 within prefix, omit EQUIV, permute TORO information to g2o"}
    # Refuse both existing paths before writing either one.
    sidecar = output.with_suffix(output.suffix + ".provenance.json")
    if output.exists() or sidecar.exists():
        raise FileExistsError("output or provenance already exists")
    with output.open("xb") as stream:
        stream.write(encoded)
    with sidecar.open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
        stream.write("\n")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vertices", type=int, required=True)
    parser.add_argument("--between-only", action="store_true", help="explicitly omit EQUIV; benchmark a derived between-factor subgraph")
    args = parser.parse_args()
    print(json.dumps(convert(args.input, args.output, args.vertices, args.between_only), indent=2))
