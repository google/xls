def prettify(seq, perline):
    seq = tuple(seq)
    r = ",\n".join([", ".join(seq[i:i+perline]) for i in range(0,len(seq),perline)])
    return r
