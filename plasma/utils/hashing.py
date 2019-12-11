from __future__ import print_function
import dill
import hashlib
import copy


def general_object_hash(o):
    """
    Serialize and hash a dictionary, list, tuple, or set (nested to any level),
    containing only other hashable types (including any lists, tuples, sets, &
    dictionaries). Relies on myhash(), which relies on dill for serialization

    Reference: https://stackoverflow.com/questions/5884066/hashing-a-dictionary

    Requirements:
    - Supports nested dictionaries (possibly with str keys only?)
    - Well-defined for a given object beyond its lifetime
    - Stable for multiple invocations of the Python kernel and across different
    machines
    """

    # TODO(KGF): currently, likely no unordered set/frozenset dict vals in conf
    # dictionary passed to this fn. There is no sorting in this 1st branch, so
    # resulting tuple of hashes would have arbitrary ordering...
    if isinstance(o, (set, frozenset, tuple, list)):
        # for (possibly mutable) ordered sequence types (list, tuple) and
        # unordered set types (set, frozenset): independently hash each element
        # in the container, and convert to immutable tuple
        return tuple([general_object_hash(e) for e in o])

    elif not isinstance(o, dict):
        # sequence and set types handled above, mapping types (dict) below.
        # Other objs can be directly serialized and hashed, including:
        # - Text sequence (str)
        # - Binary sequence (bytes)
        # - Numeric types
        return myhash_obj(o)

    new_o = copy.deepcopy(o)
    # recursively call this function when given a dictionary
    for k, v in new_o.items():
        # in the deep copy, replace the value of dict entry with its hash (int)
        new_o[k] = general_object_hash(v)

    # TODO(KGF): consider alternative suggested in above post. Instead of
    # sorted + frozenset + dill.dumps(), json.dumps() can do 1st + 3rd steps
    # Keys must all be strings in this method. Add explicit check of this cond.

    # import json
    # json.dumps(new_o, sort_keys=True, ensure_ascii=True) # or False?

    # With all values of the (possibly nested) dict obj replaced by integer
    # hashes or tuples of hashes, sort a list of unique (key, hash) items, and
    # convert to immutable tuple obj before passing to serialize+hash method
    return myhash_obj(tuple(sorted(frozenset(new_o.items()))))
    # TODO(KGF): sorted() is stable, so it won't swap relative order of
    # elements that compare equal. However, if the initial aribtrary ordering
    # changes between kernel invocations, then two "equal" items will remain
    # reordered after sorted(), thus changing the final hash. Probably fine if
    # the lowest level of hashed object in nested struct avoids hash collisions


def myhash_obj(x):
    '''
    Serialize a generic Python object using dill, decode the bytes obj,
    then pass the Unicode string to the particular hash function.
    '''
    return myhash(dill.dumps(x).decode('unicode_escape'))


def myhash_signals(signals):
    '''
    Given a List of Signal class instances, sort by their str representations
    (descriptions), concatenate their hexadecimal hashes (converted to
    base-10), and hash the resulting str
    '''
    return myhash(''.join(tuple(map(lambda x: "{}".format(x.__hash__()),
                                    sorted(signals)))))


def myhash(x):
    '''
    Hash a str with MD5 and return as a decimal (integer)

    hashlib is used instead of Python built-in method hash()
    in order to guarantee a stable/well-defined hash algorithm

    Since Python 3.3, PYTHONHASHSEED=random is the default setting, which
    randomizes the seed used for salting the hash() function to prevent DoS
    attacks that exploit hash collisions.

    See http://ocert.org/advisories/ocert-2011-003.html.

    Python 3.4 further improves the default hash algorithm (PEP 456).
    '''
    # re-encode the string into bytes object with UTF-8, create hash class obj
    # using MD5 algorithm, then return hex digits as str type, which finally
    # int(..., 16) ---> convert hexadecimal hash to base-10 integer
    return int(hashlib.md5(x.encode('utf-8')).hexdigest(), 16)
