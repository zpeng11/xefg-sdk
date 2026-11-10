"""Join public resources and package-owned metadata at the executor boundary."""


def bind_inputs(variant, resources, plan):
    names = set(variant["inputs"])
    if set(resources) != names:
        raise ValueError("public input set does not match model variant")
    if names.intersection(plan.feeds):
        raise ValueError("profile metadata must not replace application resources")
    return {**plan.feeds, **resources}
