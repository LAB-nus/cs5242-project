import json

def parseProjectJson():
    objectData = None
    relationshipData = None
    with open('../object1_object2.json') as f:
        objectData = json.load(f)
        f.close()
    
    with open('../relationship.json') as f:
        relationshipData = json.load(f)
        f.close()
    return objectData, relationshipData