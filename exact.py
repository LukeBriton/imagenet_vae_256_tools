STRICT_SHAPENET_TO_IMAGENET = {
  "02691156 airplane": [
    (404, "airliner"),
    (895, "warplane"),
  ],

  "02747177 trash bin": [
    (412, "ashcan, trash can, garbage can, wastebin"),
  ],

  "02773838 bag": [
    (414, "backpack, back pack, knapsack, packsack, rucksack"),
    (636, "mailbag, postbag"),
    (728, "plastic bag"),
    (748, "purse"),
  ],

  "02801938 basket": [
    (790, "shopping basket"),
  ],

  "02808440 bathtub": [
    (435, "bathtub, bathing tub, bath, tub"),
  ],

  "02818832 bed": [
    (564, "four-poster"),
    (520, "crib, cot"),
  ],

  "02828884 bench": [
    (703, "park bench"),
  ],

  "02834778 bicycle": [
    (444, "bicycle-built-for-two, tandem bicycle, tandem"),
    (671, "mountain bike, all-terrain bike, off-roader"),
  ],

  "02843684 birdhouse": [
    (448, "birdhouse"),
  ],

  "02871439 bookshelf": [
    (453, "bookcase"),
  ],

  "02876657 bottle": [
    (440, "beer bottle"),
    (737, "pop bottle, soda bottle"),
    (898, "water bottle"),
    (907, "wine bottle"),
  ],

  "02880940 bowl": [
    (659, "mixing bowl"),
    (809, "soup bowl"),
  ],

  "02924116 bus": [
    (779, "school bus"),
    (654, "minibus"),
  ],

  "02933112 cabinet": [
    (495, "china cabinet, china closet"),
    (648, "medicine chest, medicine cabinet"),
    (894, "wardrobe, closet, press"),
  ],

  "02942699 camera": [
    (732, "Polaroid camera, Polaroid Land camera"),
    (759, "reflex camera"),
  ],

  # NOTE: strict-only drops 02946921 can, because ImageNet has things like "milk can"
  # but not a clean generic "tin can" class that matches ShapeNet's usual "can" category.
  # (If you want to include "milk can" as a *near-strict* subtype, say so.)

  "02958343 car": [
    (609, "jeep, landrover"),
    (817, "sports car, sport car"),
    (705, "passenger car, coach, carriage"),
    (751, "racer, race car, racing car"),
    (468, "cab, hack, taxi, taxicab"),
    (511, "convertible"),
    (627, "limousine, limo"),
    (717, "pickup, pickup truck"),
  ],

  "03001627 chair": [
    (559, "folding chair"),
    (765, "rocking chair, rocker"),
    (423, "barber chair"),
  ],

  "03085013 keyboard": [
    (508, "computer keyboard, keypad"),
    (878, "typewriter keyboard"),
  ],

  "03207941 dishwasher": [
    (534, "dishwasher, dish washer, dishwashing machine"),
  ],

  "03211117 display": [
    (664, "monitor"),
    (782, "screen, CRT screen"),
    (851, "television, television system"),
  ],

  "03337140 file cabinet": [
    (553, "file, file cabinet, filing cabinet"),
  ],

  "03467517 guitar": [
    (402, "acoustic guitar"),
    (546, "electric guitar"),
  ],

  "03636649 lamp": [
    (846, "table lamp"),
    (619, "lampshade, lamp shade"),
  ],

  "03642806 laptop": [
    (620, "laptop, laptop computer"),
  ],

  "03691459 loudspeaker": [
    (632, "loudspeaker, speaker, speaker unit"),
  ],

  "03710193 mailbox": [
    (637, "mailbox, letter box"),
  ],

  "03759954 microphone": [
    (650, "microphone, mike"),
  ],

  "03790512 motorbike": [
    (670, "motor scooter, scooter"),
    (665, "moped"),
  ],

  "03797390 mug": [
    (504, "coffee mug"),
  ],

  "03928116 piano": [
    (579, "grand piano, grand"),
    (881, "upright, upright piano"),
  ],

  "03938244 pillow": [
    (721, "pillow"),
  ],

  "03948459 pistol": [
    (763, "revolver, six-gun, six-shooter"),
  ],

  "04004475 printer": [
    (742, "printer"),
  ],

  "04074963 remote": [
    (761, "remote control, remote"),
  ],

  "04090263 rifle": [
    (764, "rifle"),
    (413, "assault rifle, assault gun"),
  ],

  "04256520 sofa": [
    (831, "studio couch, day bed"),
  ],

  "04330267 stove": [
    (827, "stove"),
  ],

  "04379243 table": [
    (532, "dining table, board"),
    (736, "pool table, billiard table, snooker table"),
  ],

  "04401088 telephone": [
    (487, "cellular telephone, cellular phone, cellphone, mobile phone"),
    (528, "dial telephone, dial phone"),
    (707, "pay-phone, pay-station"),
  ],

  "04460130 tower": [
    (900, "water tower"),
  ],

  "04468005 train": [
    (466, "bullet train, bullet"),
    (547, "electric locomotive"),
    (820, "steam locomotive"),
    (829, "streetcar, tram, tramcar, trolley"),
  ],

  "04530566 watercraft": [
    (472, "canoe"),
    (484, "catamaran"),
    (510, "container ship, containership, container vessel"),
    (554, "fireboat"),
    (576, "gondola"),
    (625, "lifeboat"),
    (780, "schooner"),
    (814, "speedboat"),
    (833, "submarine, U-boat"),
    (871, "trimaran"),
    (914, "yawl"),
  ],

  "04554684 washer": [
    (897, "washer, automatic washer, washing machine"),
  ],
}

STRICT_KEEP_IMAGENET_IDS = sorted({i for v in STRICT_SHAPENET_TO_IMAGENET.values() for (i, _) in v})
print(STRICT_KEEP_IMAGENET_IDS)