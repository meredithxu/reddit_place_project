# Not all the projects on the final canvas are labeled in the atlas
# Create a new file: atlas_complete.json
import json

with open("../data/atlas.json") as f:
    data = json.load(f)

with open("../data/new_projects.json") as f:
    new_projects = json.load(f)

max_id = 0

print("Orignal atlas length: ", len(data["atlas"]))
print("Number of new projects: ", len(new_projects["atlas"]))
for project in data["atlas"]:
    project_id = int(project["id"])
    if project_id > max_id:
        max_id = project_id

for project in new_projects["atlas"]:
    max_id += 1
    project["id"] = max_id

    data["atlas"].append(project)

with open('../data/atlas_complete.json', 'w') as fp:
    json.dump(data, fp, indent=4)



print("Atlas complete length: ", len(data["atlas"]) )