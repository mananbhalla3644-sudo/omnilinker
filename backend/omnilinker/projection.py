"""Graph projection (blueprint 5.2).

**The graph is a derived store.** It is a pure function of the documents, which
is what makes the whole identity pipeline safe: re-resolving identities and
re-projecting the graph is always available, and a bug in either can be
repaired by dropping and rebuilding rather than by surgery on live data.

That is why edges carry provenance:

    (a)-[:LINKED_TO {state:"advisory", detector:"D3", confidence:0.4}]->(b)

`state` is the privacy contract from 11.5. An advisory edge is a machine's
suggestion. Only a user action may write `state:"accepted"`, and
`accept_insight()` is the only function in the codebase that does it.

Node/edge model
---------------
    Workspace
    Person          resolved human
    Identity        per-provider account (Person -[:HAS_IDENTITY]-> Identity)
    Conversation    slack channel, gmail thread, whatsapp chat, notion space
    Message         includes emails
    File, Note, Video, Transcript
    Event           a deadline extracted from a message

    (Person)  -[:HAS_IDENTITY]->    (Identity)
    (Message) -[:SENT_BY]->         (Person)
    (Person)  -[:PARTICIPATED_IN]-> (Conversation)
    (Message) -[:IN_CONVERSATION]-> (Conversation)
    (Message) -[:REPLY_TO]->        (Message)
    (Message) -[:THREAD_ROOT]->     (Message)
    (Message) -[:ATTACHED_TO]->     (File)
    (Message) -[:MENTIONS]->        (Person)
    (Message) -[:PREDICTED_DUE]->   (Event)
    (File)    -[:OWNS]->            (Person)
    (File)    -[:ABOUT]->           (Person)
    (Note)    -[:AUTHORED]->        (Person)
    (Note)    -[:PART_OF]->         (Note)
    (Video)   -[:HAS_TRANSCRIPT]->  (Transcript)
    (Person)  -[:LINKED_TO]->       (Person)   state: advisory
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

from omnilinker.crypto.tokens import token_for_email
from omnilinker.ids import prefixed
from omnilinker.store import get_stores
from omnilinker.store.base import Edge, Node, Query

MENTION_LABEL = re.compile(r"^@\w[\w.\-]*$")
PERSON_NODE_LIMIT = 200_000


class GraphProjector:
    """Rebuild or incrementally update the relationship graph from documents."""

    def __init__(self) -> None:
        self.stores = get_stores()

    # ------------------------------------------------------------------
    def rebuild(self, workspace: str) -> dict[str, Any]:
        """Full rebuild. Purges and re-projects every edge and node.

        This is the operation you reach for after an identity re-resolution, and
        it is cheap enough to run on a schedule: it reads documents, not the
        providers.
        """
        t0 = _now_ms()
        removed = self.stores.graph.purge()
        stats = self.project(workspace)
        stats.update({"purged_nodes": removed, "mode": "rebuild",
                      "duration_ms": _now_ms() - t0})
        return stats

    def project(self, workspace: str) -> dict[str, Any]:
        docs = self.stores.docs
        identities = docs.find(Query("identities", filters=[], limit=PERSON_NODE_LIMIT))
        persons = docs.find(Query("persons", limit=PERSON_NODE_LIMIT))
        messages = docs.find(Query("messages", sort=[("ts", 1)], limit=PERSON_NODE_LIMIT))
        files = docs.find(Query("files", limit=PERSON_NODE_LIMIT))
        notes = docs.find(Query("notes", limit=PERSON_NODE_LIMIT))
        videos = docs.find(Query("videos", limit=PERSON_NODE_LIMIT))
        transcripts = docs.find(Query("transcripts", limit=PERSON_NODE_LIMIT))
        conversations = docs.find(Query("conversations", limit=PERSON_NODE_LIMIT))

        ref_to_person: dict[str, str] = {}
        for identity in identities:
            pid = identity.get("person_id")
            ref = identity.get("ref")
            if pid and ref:
                ref_to_person.setdefault(str(ref), str(pid))

        nodes: list[Node] = []
        edges: list[Edge] = []

        # -- Workspace ---------------------------------------------------
        nodes.append(Node(id=f"ws_{workspace}", label="Workspace",
                          props={"workspace_id": workspace, "name": "My OmniLinker"}))

        # -- Persons + Identities ----------------------------------------
        for person in persons:
            pid = str(person.get("person_id"))
            nodes.append(Node(id=pid, label="Person", props={
                "display_name": person.get("display_name", "Unknown"),
                "providers": person.get("providers", []),
                "message_count": person.get("message_count", 0),
                "cross_source": person.get("cross_source", False),
                "is_self": person.get("is_self", False),
                "email_present": bool(person.get("email")),
                "last_seen": person.get("last_seen", ""),
            }))
        for identity in identities:
            iid = str(identity.get("_id"))
            if iid not in {str(i.get("_id")) for i in identities}:  # pragma: no cover
                continue
            nodes.append(Node(id=iid, label="Identity", props={
                "provider": identity.get("provider", ""),
                "display_name": identity.get("display_name", ""),
                "mention_count": identity.get("mention_count", 0),
                "kind": "contact" if str(identity.get("ref", "")).startswith("contact:")
                        else "human",
                "email_token": identity.get("email_token", ""),
            }))
            pid = identity.get("person_id")
            if pid and not str(identity.get("ref", "")).startswith("contact:"):
                edges.append(Edge(type="HAS_IDENTITY", src=str(pid), dst=iid))

        # -- Conversations ------------------------------------------------
        conv_key_to_id: dict[str, str] = {}
        for conv in conversations:
            cid = str(conv.get("_id"))
            conv_key_to_id[str(conv.get("conversation_key") or cid)] = cid
            nodes.append(Node(id=cid, label="Conversation", props={
                "provider": conv.get("provider", ""),
                "label": conv.get("label", ""),
                "kind": conv.get("kind", ""),
                "message_count": conv.get("message_count", 0),
                "first_ts": conv.get("first_ts", ""),
                "last_ts": conv.get("last_ts", ""),
            }))

        # -- Messages -----------------------------------------------------
        participants: dict[str, set[str]] = defaultdict(set)
        for msg in messages:
            mid = str(msg.get("_id"))
            nodes.append(Node(id=mid, label="Message", props={
                "provider": msg.get("provider", ""),
                "ts": msg.get("ts", ""),
                "thread_root_id": msg.get("thread_root_id", ""),
                "importance": msg.get("importance", 0.0),
                "attachment_count": len(msg.get("attachments") or []),
                "deadline_count": len((msg.get("entities") or {}).get("deadlines") or []),
            }))
            sender_ref = str(msg.get("sender_ref") or "")
            sender_person = ref_to_person.get(sender_ref)
            if sender_person:
                edges.append(Edge(type="SENT_BY", src=mid, dst=sender_person))
            elif sender_ref:
                # Unresolved sender: keep the edge to a synthetic Person so the
                # message is not an orphan in the graph. A dangling "Unknown"
                # node is honest; a silently missing message is not.
                placeholder = f"per_unknown_{_slug(sender_ref)}"
                nodes.append(Node(id=placeholder, label="Person",
                                  props={"display_name": sender_ref, "unresolved": True}))
                edges.append(Edge(type="SENT_BY", src=mid, dst=placeholder))

            conv_id = conv_key_to_id.get(f"{msg.get('provider')}:{msg.get('stream')}") \
                or conv_key_to_id.get(f"{msg.get('provider')}:{msg.get('conversation_id')}")
            if conv_id:
                edges.append(Edge(type="IN_CONVERSATION", src=mid, dst=conv_id))
                if sender_ref:
                    participants[conv_id].add(sender_ref)
                # Person -> Conversation, from the sender identity
                if sender_person:
                    edges.append(Edge(type="PARTICIPATED_IN", src=sender_person, dst=conv_id))

            root = str(msg.get("thread_root_id") or "")
            reply_to = str(msg.get("reply_to_id") or "")
            if reply_to and reply_to != mid:
                edges.append(Edge(type="REPLY_TO", src=mid, dst=_as_msg_id(msg, reply_to)))
            elif root and root != mid:
                edges.append(Edge(type="THREAD_ROOT", src=mid, dst=_as_msg_id(msg, root)))

            for att in msg.get("attachments") or []:
                fid = att.get("provider_file_id")
                if fid:
                    edges.append(Edge(type="ATTACHED_TO", src=mid,
                                      dst=_file_id(msg.get("provider", ""), fid)))

            for mention in _mentioned_refs(msg, ref_to_person):
                edges.append(Edge(type="MENTIONS", src=mid, dst=mention))

            for deadline in (msg.get("entities") or {}).get("deadlines") or []:
                event_id = f"evt_{mid}_{abs(hash(deadline.get('due',''))) % 10**8}"
                nodes.append(Node(id=event_id, label="Event", props={
                    "due": deadline.get("due", ""),
                    "hardness": deadline.get("hardness", "medium"),
                    "cue": deadline.get("cue", ""),
                    "confidence": deadline.get("confidence", 0.0),
                    "state": "advisory",
                }))
                edges.append(Edge(type="PREDICTED_DUE", src=mid, dst=event_id,
                                  props={"state": "advisory"}))

        # -- Files ---------------------------------------------------------
        for f in files:
            fid = str(f.get("_id"))
            nodes.append(Node(id=fid, label="File", props={
                "provider": f.get("provider", ""),
                "name": f.get("name", ""),
                "mime": f.get("mime", ""),
                "size_bytes": f.get("size_bytes", 0),
                "modified_ts": f.get("modified_ts", ""),
                "folder_path": f.get("folder_path", []),
            }))
            owner_ref = str(f.get("owner_ref") or "")
            owner_person = ref_to_person.get(owner_ref) or f.get("person_id")
            if owner_person:
                edges.append(Edge(type="OWNS", src=fid, dst=str(owner_person)))

        # -- Notes ---------------------------------------------------------
        for note in notes:
            nid = str(note.get("_id"))
            nodes.append(Node(id=nid, label="Note", props={
                "provider": note.get("provider", ""),
                "title": note.get("title", ""),
                "url": note.get("url", ""),
                "last_edited": note.get("last_edited", ""),
                "block_count": note.get("block_count", 0),
            }))
            author = note.get("person_id") or note.get("author_ref")
            author_person = ref_to_person.get(str(author or ""))
            if author_person:
                edges.append(Edge(type="AUTHORED", src=nid, dst=author_person))
            parent = str(note.get("parent_id") or "")
            if parent:
                edges.append(Edge(type="PART_OF", src=nid, dst=f"not_{_slug(parent)}"))

        # -- Videos / transcripts -----------------------------------------
        for video in videos:
            vid = str(video.get("_id"))
            nodes.append(Node(id=vid, label="Video", props={
                "provider": video.get("provider", ""),
                "title": video.get("title", ""),
                "channel_title": video.get("channel_title", ""),
                "duration_s": video.get("duration_s", 0),
                "published_ts": video.get("published_ts", ""),
                "tags": video.get("tags", []),
            }))
        for tr in transcripts:
            tid = str(tr.get("_id"))
            nodes.append(Node(id=tid, label="Transcript", props={
                "language": tr.get("language", ""),
                "seg_count": tr.get("seg_count", 0),
                "video_id": tr.get("video_id", ""),
            }))
            parent = str(tr.get("_video_id") or "")
            if parent:
                # `_video_id` is the provider's video id; the video document id is
                # `<provider>:<that id>`. Prefixing here keeps the edge pointing
                # at a node that exists instead of at a dangling string.
                edges.append(Edge(
                    type="HAS_TRANSCRIPT",
                    src=f"{tr.get('provider', 'youtube')}:{parent}",
                    dst=tid,
                ))

        # De-duplicate edges: `upsert_batch` keys on (src, type, dst), but
        # building the list without a set wastes memory on busy workspaces.
        deduped: dict[tuple[str, str, str], Edge] = {}
        for edge in edges:
            key = edge.key
            existing = deduped.get(key)
            if existing is None:
                deduped[key] = edge
            else:
                # Keep the strongest props rather than the last write.
                for k, v in edge.props.items():
                    existing.props.setdefault(k, v)

        node_seen: dict[str, Node] = {}
        for node in nodes:
            node_seen.setdefault(node.id, node)

        nodes_written, edges_written = self.stores.graph.upsert_batch(
            list(node_seen.values()), list(deduped.values())
        )
        return {
            "nodes": nodes_written,
            "edges": edges_written,
            "persons": len(persons),
            "identities": len(identities),
            "messages": len(messages),
            "files": len(files),
            "notes": len(notes),
            "conversations": len(conversations),
            "mode": "incremental",
        }

    # ------------------------------------------------------------------
    def write_insight_edges(self, insights: Iterable[Any]) -> int:
        """Materialize detected insights as ADVISORY graph edges.

        The `state: "advisory"` marker is not decoration. `write_insight_edges`
        is the only writer of Person-Person edges, and it hard-codes the state;
        `accept_insight` is the only reader that may flip it. That asymmetry is
        the mechanism, not an accident of the code layout.
        """
        count = 0
        for insight in insights:
            data = insight.to_dict() if hasattr(insight, "to_dict") else dict(insight)
            entities = [str(e) for e in data.get("entities", [])]
            if len(entities) != 2:
                continue
            if not (self.stores.graph.get_node(entities[0])
                    and self.stores.graph.get_node(entities[1])):
                continue
            self.stores.graph.upsert_edge(Edge(
                type="LINKED_TO",
                src=entities[0], dst=entities[1],
                props={
                    "state": "advisory",
                    "detector": data.get("detector", ""),
                    "confidence": data.get("confidence", 0.0),
                    "insight_id": data.get("_id", ""),
                    "kind": data.get("kind", ""),
                },
            ))
            count += 1
        return count

    def accept_insight(self, insight_id: str, insight_kind: str,
                       source_id: str, target_id: str) -> bool:
        """Flip one advisory edge to accepted. **User-action-only.**

        Called from exactly one place: the API route that handles an explicit
        user confirmation. There is no automatic path here, and that is the
        point - the blueprint's rule is that the system may suggest a link but
        only a person may assert it.
        """
        if not (self.stores.graph.get_node(source_id) and self.stores.graph.get_node(target_id)):
            return False
        self.stores.graph.delete_edge(source_id, "LINKED_TO", target_id)
        self.stores.graph.upsert_edge(Edge(
            type="LINKED_TO", src=source_id, dst=target_id,
            props={"state": "accepted", "detector": insight_kind, "insight_id": insight_id,
                   "accepted_by": "user"},
        ))
        return True

    def dismiss_insight(self, source_id: str, target_id: str) -> bool:
        self.stores.graph.delete_edge(source_id, "LINKED_TO", target_id)
        return True


# --------------------------------------------------------------------------


def _mentioned_refs(msg: Mapping[str, Any], ref_to_person: Mapping[str, str]) -> list[str]:
    """@handles in the body resolve to people when we know the handle.

    Only handles we have an identity for are linked. Inventing a Person for an
    unknown handle would fill the graph with stubs and make the view useless.
    """
    out: list[str] = []
    text = msg.get("body_text")
    if isinstance(text, Mapping):  # sealed, skip - the graph never needs content
        return out
    for raw in re.findall(r"@([A-Za-z0-9._\-]{2,30})", str(text or "")):
        for provider in ("slack", "discord", "gmail", "whatsapp"):
            pid = ref_to_person.get(f"{provider}:{raw}")
            if pid:
                out.append(pid)
                break
    return out


def _as_msg_id(msg: Mapping[str, Any], provider_id: str) -> str:
    """Thread roots and reply targets arrive as provider ids; the graph is keyed
    by our own document ids."""
    return f"{msg.get('provider', '')}:{provider_id}"


def _file_id(provider: str, provider_file_id: str) -> str:
    return f"{provider}:{provider_file_id}"


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(value or ""))[:64] or "x"


def _now_ms() -> int:
    import time

    return int(time.time() * 1000)
