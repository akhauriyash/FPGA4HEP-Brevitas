import yaml

def parse_config(config_file) :
    print("Loading configuration from", config_file)
    config = open(config_file, 'r')
    return yaml.load(config)

def counter(vector):
    s = vector.shape[0]
    k = []
    v = vector.detach()
    v[v!=0] = 1
    for i in range(s):
        k.append(torch.sum(v[i, :]).item())
    return k

def counter_fanout(vector):
    s = vector.shape[1]
    k = []
    v = vector.detach()
    v[v!=0] = 1
    for i in range(s):
        k.append(torch.sum(v[:, i]).item())
    return k
