import os
from typing import List, Optional, Set, Tuple

import pyshacl
from rdflib import RDF, Graph, Literal, Namespace, URIRef

import paths
from testcase import WorldPoint

# Resolved against the repo, not the working directory, so the grader can be run
# from anywhere.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Fixed here, never read back out of the submitted file. A namespace typo has
# to be reported, and a prefix read from the record would silently follow it.
HW2 = Namespace("http://taica.course/hw2/ontology#")
SH = Namespace("http://www.w3.org/ns/shacl#")


def load_record(case_id: str, part1_output: str = paths.PART1_OUTPUT) -> Graph:
    """Parse one case's record: instances only, with no shapes attached."""
    return Graph().parse(os.path.join(part1_output, case_id, paths.RECORD_NAME),
                         format="turtle")


def load_rules(path: str = None) -> Graph:
    """The vocabulary and the feasibility rules a planner declared, in one graph."""
    return (Graph().parse(os.path.join(REPO, paths.ONTOLOGY_SCHEMA), format="turtle")
            .parse(path or os.path.join(REPO, paths.ONTOLOGY_RULES), format="turtle"))


def load_structure(path: str = None) -> Graph:
    """The structural shapes every record must satisfy, whatever rules it declared."""
    return Graph().parse(path or os.path.join(REPO, paths.ONTOLOGY_STRUCTURE),
                         format="turtle")


NOT_A_CHAIN = ("the edges typed hw2:PathEdge do not form one chain running "
               "from the run's root node to its goal node")


def _name(node) -> str:
    """An individual's local name, such as `edge_12`: the part a reader acts on."""
    return str(node).rsplit("#", 1)[-1].rsplit("/", 1)[-1]


def _by_number(name: str) -> Tuple[str, int]:
    head, _, tail = name.rpartition("_")
    return (head, int(tail)) if tail.isdigit() else (name, -1)


def _reported(graph: Graph, shapes: Graph) -> List[Tuple[URIRef, str]]:
    """Every (focus node, message) the shapes report against this graph."""
    _conforms, results, _text = pyshacl.validate(graph, shacl_graph=shapes)
    return [(results.value(result, SH.focusNode),
             str(results.value(result, SH.resultMessage)))
            for result in results.objects(None, SH.result)]


def violations(graph: Graph, shapes: Graph,
               focus: Set = None) -> List[Tuple[str, List[str]]]:
    """Every violation the shapes report, grouped by message and named.

    Without `focus` this includes the rejected candidates, which are meant to
    violate; pass the edges the planner kept.
    """
    grouped = {}
    for node, message in _reported(graph, shapes):
        if focus is None or node in focus:
            grouped.setdefault(message, set()).add(_name(node))
    return [(message, sorted(names, key=_by_number))
            for message, names in sorted(grouped.items())]


def typed(graph: Graph, rdf_class: URIRef) -> Set[URIRef]:
    """Every individual the record types as this class."""
    return set(graph.subjects(RDF.type, rdf_class))


def named(graph: Graph, rdf_class: URIRef) -> Set[str]:
    """The local names of every individual the record types as this class."""
    return {_name(node) for node in graph.subjects(RDF.type, rdf_class)}


def rejected_edges(graph: Graph, shapes: Graph) -> Set[URIRef]:
    """Every edge these shapes report against."""
    return {node for node, _message in _reported(graph, shapes)}


def runs(graph: Graph) -> List[URIRef]:
    """Every planning run the record holds, in the fixed namespace."""
    return sorted(graph.subjects(RDF.type, HW2.PlanningRun))


def run_of(graph: Graph) -> Optional[URIRef]:
    """The record's planning run, or None if it holds none in the fixed namespace."""
    return graph.value(predicate=RDF.type, object=HW2.PlanningRun)


def goal_object(graph: Graph) -> str:
    """The semantic category the run says it was sent to."""
    return str(graph.value(run_of(graph), HW2.goalObject))


def node_world(graph: Graph, node: URIRef) -> WorldPoint:
    """One node's world coordinates, in metres."""
    return float(graph.value(node, HW2.worldX)), float(graph.value(node, HW2.worldZ))


def goal_world(graph: Graph) -> WorldPoint:
    """The world point the run planned towards."""
    return node_world(graph, graph.value(run_of(graph), HW2.hasGoalNode))


def path_chain(graph: Graph) -> Tuple[list, str]:
    """The claimed path root-to-tail, or ([], why it is not one).

    Whether a set of edges forms one connected chain is the property SHACL Core
    cannot express, so it is code.
    """
    marked = typed(graph, HW2.PathEdge)
    if not marked:
        return [], "no edge is typed hw2:PathEdge, so no path is claimed"

    by_start = {}
    for edge in marked:
        start = graph.value(edge, HW2.hasStartNode)
        if start in by_start:
            return [], NOT_A_CHAIN
        by_start[start] = edge

    node = graph.value(run_of(graph), HW2.hasRootNode)
    chain, visited = [], {node}
    while node in by_start:
        chain.append(by_start.pop(node))
        node = graph.value(chain[-1], HW2.hasEndNode)
        if node in visited:
            return [], NOT_A_CHAIN
        visited.add(node)

    if not chain or by_start or node != graph.value(run_of(graph), HW2.hasGoalNode):
        return [], NOT_A_CHAIN
    return chain, ""


def world_path(graph: Graph, chain: list) -> List[WorldPoint]:
    """The claimed path as waypoints: the root node, then each edge's end node."""
    nodes = [graph.value(chain[0], HW2.hasStartNode)] if chain else []
    nodes += [graph.value(edge, HW2.hasEndNode) for edge in chain]
    return [node_world(graph, node) for node in nodes]


def edge_world(graph: Graph, edge: URIRef) -> List[WorldPoint]:
    """One edge as its two endpoints, in world metres."""
    return [node_world(graph, graph.value(edge, link))
            for link in (HW2.hasStartNode, HW2.hasEndNode)]


def edge_claims(graph: Graph, edge: URIRef, predicate: URIRef) -> Set[str]:
    """The names this edge claims for one of the entered-region properties."""
    return {_name(node) for node in graph.objects(edge, predicate)}


def region_claims(graph: Graph) -> List[Tuple[str, bool]]:
    """Each restricted zone the record names, and whether it claims the goal is
    inside it."""
    return sorted((_name(region),
                   Literal(True) in graph.objects(region, HW2.containsGoal))
                  for region in graph.subjects(RDF.type, HW2.RestrictedZone))


def region_individual(graph: Graph, region_id: str) -> Optional[URIRef]:
    """The individual a record uses for one test-case region id.

    The IRI's local name is that id: the record's own vocabulary carries no
    separate identifier property for a zone or a person.
    """
    for region in graph.subjects(RDF.type, HW2.RestrictedZone):
        if _name(region) == region_id:
            return region
    return None
