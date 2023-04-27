def get_chunk_type(tok, idx_to_tag):
    # method implemented in https://github.com/guillaumegenthial/sequence_tagging/blob/master/model/data_utils.py
    """
    Args:
        tok: id of token, ex 0
        idx_to_tag: dictionary {0: "B-DRUG", ...}
    Returns:
        tuple: "B", "DRUG"
    """
    tag_name = idx_to_tag[tok]
    tag_class = tag_name.split('-')[0]
    tag_type = tag_name.split('-')[-1]
    return tag_class, tag_type

def get_chunks(seq, tags):
    # method implemented in https://github.com/guillaumegenthial/sequence_tagging/blob/master/model/data_utils.py
    # Altered because previous implementation recognized entities starting with I-tags as true
    """
        Given a sequence of tags, group entities and their position
        Args:
            seq: [4, 4, 0, 0, ...] sequence of labels
            tags: dict["O"] = 4
        Returns:
            list of (chunk_type, chunk_start, chunk_end)
        Example:
            seq = [0, 1, 4, 2]
            tags = {"B-DRUG": 0, "I-DRUG": 1, "B-AE": 2, "I-AE": 3, "O": 4}
            result = [("DRUG", 0, 1), ("AE", 3, 3)]
    """
    default = tags['O']
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if tok == default and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i - 1)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tok != default:
            tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
            if (chunk_type is None) and (tok_chunk_class == "B"):
                chunk_type, chunk_start = tok_chunk_type, i
            else:
                if tok_chunk_class == "B":
                    if chunk_type is not None:
                        chunk = (chunk_type, chunk_start, i - 1)
                        chunks.append(chunk)
                    chunk_type, chunk_start = tok_chunk_type, i
                elif tok_chunk_type != chunk_type:
                    if chunk_type is not None:
                        chunk = (chunk_type, chunk_start, i - 1)
                        chunks.append(chunk)
                    chunk_type, chunk_start = None, None
        else:
            pass

    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq) - 1)
        chunks.append(chunk)

    return chunks