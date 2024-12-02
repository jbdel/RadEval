from collections import defaultdict

leaves_mapping = {
    "No Finding": 0,
    "Lung Lesion": 1,
    "Edema": 2,
    "Pneumonia": 3,
    "Atelectasis": 4,
    "Aspiration": 5,
    "Lung collapse": 6,
    "Perihilar airspace opacity": 7,
    "Air space opacity\u2013multifocal": 8,
    "Mass/Solitary lung mass": 9,
    "Nodule/Solitary lung nodule": 10,
    "Cavitating mass with content": 11,
    "Cavitating masses": 12,
    "Emphysema": 13,
    "Fibrosis": 14,
    "Pulmonary congestion": 15,
    "Hilar lymphadenopathy": 16,
    "Bronchiectasis": 17,
    "Simple pneumothorax": 18,
    "Loculated pneumothorax": 19,
    "Tension pneumothorax": 20,
    "Simple pleural effusion": 21,
    "Loculated pleural effusion": 22,
    "Pleural scarring": 23,
    "Hydropneumothorax": 24,
    "Pleural Other": 25,
    "Cardiomegaly": 26,
    "Pericardial effusion": 27,
    "Inferior mediastinal mass": 28,
    "Superior mediastinal mass": 29,
    "Tortuous Aorta": 30,
    "Calcification of the Aorta": 31,
    "Enlarged pulmonary artery": 32,
    "Hernia": 33,
    "Pneumomediastinum": 34,
    "Tracheal deviation": 35,
    "Acute humerus fracture": 36,
    "Acute rib fracture": 37,
    "Acute clavicle fracture": 38,
    "Acute scapula fracture": 39,
    "Compression fracture": 40,
    "Shoulder dislocation": 41,
    "Subcutaneous Emphysema": 42,
    "Suboptimal central line": 43,
    "Suboptimal endotracheal tube": 44,
    "Suboptimal nasogastric tube": 45,
    "Suboptimal pulmonary arterial catheter": 46,
    "Pleural tube": 47,
    "PICC line": 48,
    "Port catheter": 49,
    "Pacemaker": 50,
    "Implantable defibrillator": 51,
    "LVAD": 52,
    "Intraaortic balloon pump": 53,
    "Pneumoperitoneum": 54
}

leaves_labels = list(leaves_mapping.keys())
########################################################################

disease_mapping = {
    "No Finding": "No Finding",
    "Lung Lesion": "Lung Finding",
    "Fibrosis": "Lung Finding",
    "Emphysema": "Lung Finding",
    "Pulmonary congestion": "Lung Finding",
    "Bronchiectasis": "Lung Finding",
    "Lung Finding": "Lung Finding",
    "Hilar lymphadenopathy": "Lung Finding",
    "Diffuse air space opacity": "Air space opacity",
    "Air space opacity–multifocal": "Air space opacity",
    "Edema": "Diffuse air space opacity",
    "Consolidation": "Focal air space opacity",
    "Focal air space opacity": "Focal air space opacity",
    "Perihilar airspace opacity": "Focal air space opacity",
    "Pneumonia": "Consolidation",
    "Atelectasis": "Consolidation",
    "Aspiration": "Consolidation",
    "Segmental collapse": "Focal air space opacity",
    "Lung collapse": "Segmental collapse",
    "Solitary masslike opacity": "Masslike opacity",
    "Mass/Solitary lung mass": "Solitary masslike opacity",
    "Nodule/Solitary lung nodule": "Solitary masslike opacity",
    "Cavitating mass with content": "Solitary masslike opacity",
    "Multiple masslike opacities": "Masslike opacity",
    "Cavitating masses": "Multiple masslike opacities",
    "Pneumothorax": "Pleural finding",
    "Hydropneumothorax": "Pleural finding",
    "Pleural Other": "Pleural finding",
    "Simple pneumothorax": "Pneumothorax",
    "Loculated pneumothorax": "Pneumothorax",
    "Tension pneumothorax": "Pneumothorax",
    "Pleural Effusion": "Pleural Thickening",
    "Pleural scarring": "Pleural Thickening",
    "Simple pleural effusion": "Pleural Effusion",
    "Loculated pleural effusion": "Pleural Effusion",
    "Widened cardiac silhouette": "None",
    "Cardiomegaly": "Widened cardiac silhouette",
    "Pericardial effusion": "Widened cardiac silhouette",
    "Hernia": "Mediastinal finding",
    "Mediastinal mass": "Mediastinal finding",
    "Pneumomediastinum": "Mediastinal finding",
    "Tracheal deviation": "Mediastinal finding",
    "Inferior mediastinal mass": "Mediastinal mass",
    "Superior mediastinal mass": "Mediastinal mass",
    "Widened aortic contour": "Vascular finding",
    "Tortuous Aorta": "Widened aortic contour",
    "Fracture": "Musculoskeletal finding",
    "Shoulder dislocation": "Musculoskeletal finding",
    "Acute humerus fracture": "Fracture",
    "Acute rib fracture": "Fracture",
    "Acute clavicle fracture": "Fracture",
    "Acute scapula fracture": "Fracture",
    "Compression fracture": "Fracture",
    "Chest wall finding": "Musculoskeletal finding",
    "Subcutaneous Emphysema": "Chest wall finding",
    "Suboptimal central line": "Support Devices",
    "Suboptimal endotracheal tube": "Support Devices",
    "Suboptimal nasogastric tube": "Support Devices",
    "Suboptimal pulmonary arterial catheter": "Support Devices",
    "Pleural tube": "Support Devices",
    "PICC line": "Support Devices",
    "Port catheter": "Support Devices",
    "Pacemaker": "Support Devices",
    "Implantable defibrillator": "Support Devices",
    "LVAD": "Support Devices",
    "Intraaortic balloon pump": "Support Devices",
    "Subdiaphragmatic gas": "Upper abdominal finding",
    "Pneumoperitoneum": "Subdiaphragmatic gas",
    "Calcification of the Aorta": "Vascular finding",
    "Enlarged pulmonary artery": "Vascular finding",
}

upper_labels = list(set(disease_mapping.values()))
upper_mapping = {label: idx for idx, label in enumerate(upper_labels)}

########################################################################

# Create a list with three versions of each label
leaves_with_statuses = [f"{label} (Present)" for label in leaves_labels if label != "No Finding"] + \
                       [f"{label} (Uncertain)" for label in leaves_labels if label != "No Finding"] + \
                       [f"{label} (Absent)" for label in leaves_labels if label != "No Finding"]
leaves_with_statuses.append('No Finding (None)')
leaves_with_statuses_mapping = {label: idx for idx, label in enumerate(leaves_with_statuses)}

# print(len(leaves_labels))  # 55
# print(len(leaves_with_statuses))  # 163

########################################################################

reversed_mapping = defaultdict(list)
for lower_disease, upper_disease in disease_mapping.items():
    reversed_mapping[upper_disease].append(lower_disease)

upper_with_statuses = [f"{label} (Present)" for label in upper_labels if label != "No Finding"] + \
                      [f"{label} (Uncertain)" for label in upper_labels if label != "No Finding"] + \
                      [f"{label} (Absent)" for label in upper_labels if label != "No Finding"]
upper_with_statuses.append('No Finding (None)')
upper_with_statuses_mapping = {label: idx for idx, label in enumerate(upper_with_statuses)}

# print(len(reversed_mapping))  # 26
# print(len(upper_with_statuses))  # 76

########################################################################
