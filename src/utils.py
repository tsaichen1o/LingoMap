"""
LingoMap Utils
"""

from rdflib import Graph, URIRef
from rdflib.namespace import NamespaceManager

def compact_uri(full_uri: str, ns_manager: NamespaceManager) -> str:
    """
    Try to shorten a full URI to a prefixed name (QName).
    If no corresponding prefix is found, return the full URI wrapped in angle brackets.
    """
    try:
        # rdflib's qname() method will automatically do this work
        prefixed_name = ns_manager.qname(URIRef(full_uri))
        return prefixed_name
    except:
        # If it cannot be shortened, return the original format
        return f"<{full_uri}>"