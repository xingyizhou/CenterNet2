# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import os

from fvcore.common.timer import Timer
from detectron2.structures import BoxMode
from fvcore.common.file_io import PathManager
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.lvis_v1_categories import LVIS_CATEGORIES

logger = logging.getLogger(__name__)

__all__ = ["load_lvis_v1_json", "register_lvis_v1_instances", "get_lvis_v1_instances_meta"]


def register_lvis_v1_instances(name, metadata, json_file, image_root):
    """
    Register a dataset in LVIS's json annotation format for instance detection and segmentation.

    Args:
        name (str): a name that identifies the dataset, e.g. "lvis_v0.5_train".
        metadata (dict): extra metadata associated with this dataset. It can be an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str): directory which contains all the images.
    """
    DatasetCatalog.register(name, lambda: load_lvis_v1_json(json_file, image_root, name))
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="lvis", **metadata
    )


def load_lvis_v1_json(json_file, image_root, dataset_name=None):
    from lvis import LVIS

    json_file = PathManager.get_local_path(json_file)

    timer = Timer()
    lvis_api = LVIS(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    if dataset_name is not None:
        meta = get_lvis_v1_instances_meta()
        MetadataCatalog.get(dataset_name).set(**meta)

    img_ids = sorted(lvis_api.imgs.keys())
    imgs = lvis_api.load_imgs(img_ids)
    anns = [lvis_api.img_ann_map[img_id] for img_id in img_ids]

    ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
    assert len(set(ann_ids)) == len(ann_ids), "Annotation ids in '{}' are not unique".format(
        json_file
    )

    imgs_anns = list(zip(imgs, anns))

    logger.info("Loaded {} images in the LVIS v1 format from {}".format(len(imgs_anns), json_file))

    dataset_dicts = []

    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        if "file_name" in img_dict:
            file_name = img_dict["file_name"]
            if img_dict["file_name"].startswith("COCO"):
                # Convert form the COCO 2014 file naming convention of
                # COCO_[train/val/test]2014_000000000000.jpg to the 2017 naming convention of
                # 000000000000.jpg (LVIS v1 will fix this naming issue)
                file_name = file_name[-16:]
        else:
            # e.g., http://images.cocodataset.org/train2017/000000391895.jpg
            file_name = img_dict["coco_url"][30:]
        
        record["file_name"] = os.path.join(image_root, file_name)
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        record["not_exhaustive_category_ids"] = img_dict.get("not_exhaustive_category_ids", [])
        record["neg_category_ids"] = img_dict.get("neg_category_ids", [])
        # NOTE: modified by zhouxy to convert to 0-based
        record["neg_category_ids"] = [x - 1 for x in record["neg_category_ids"]]
        image_id = record["image_id"] = img_dict["id"]

        objs = []
        for anno in anno_dict_list:
            # Check that the image_id in this annotation is the same as
            # the image_id we're looking at.
            # This fails only when the data parsing logic or the annotation file is buggy.
            assert anno["image_id"] == image_id
            if anno.get('iscrowd', 0) > 0:
                continue
            obj = {"bbox": anno["bbox"], "bbox_mode": BoxMode.XYWH_ABS}
            obj["category_id"] = anno["category_id"] - 1  # Convert 1-indexed to 0-indexed
            # segm = anno["segmentation"]  # list[list[float]]
            # filter out invalid polygons (< 3 points)
            # valid_segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
            # assert len(segm) == len(
            #     valid_segm
            # ), "Annotation contains an invalid polygon with < 3 points"
            # if not len(segm) == len(valid_segm):
            #   print('Annotation contains an invalid polygon with < 3 points')
            # assert len(segm) > 0
            # obj["segmentation"] = segm
            obj['score'] = anno['score'] if 'score' in anno else 1.
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts


def get_lvis_v1_instances_meta():
    assert len(LVIS_CATEGORIES) == 1203
    cat_ids = [k["id"] for k in LVIS_CATEGORIES]
    assert min(cat_ids) == 1 and max(cat_ids) == len(
        cat_ids
    ), "Category ids are not in [1, #categories], as expected"
    # Ensure that the category list is sorted by id
    lvis_categories = sorted(LVIS_CATEGORIES, key=lambda x: x["id"])
    thing_classes = [k["synonyms"][0] for k in lvis_categories]
    meta = {"thing_classes": thing_classes}
    return meta

_PREDEFINED_SPLITS_LVIS_V1 = {
    # "lvis_v1_train": ("coco/", "lvis/lvis_v1_train.json"),
    # "lvis_v1_val": ("coco/", "lvis/lvis_v1_val.json"),
    # "lvis_v1_test-dev": ("coco/", "lvis/lvis_v1_image_info_test_dev.json"),
    # "lvis_v1_test-challenge": ("coco/", "lvis/lvis_v1_image_info_test_challenge.json"),
    "lvis_v1_val_allneg": ("coco/", "lvis/lvis_v1_val_allneg.json"),
    "lvis_v1_train+coco_bug": ("coco/", "lvis/lvis_v1_train+coco.json"),
    "lvis_v1_train+coco": ("coco/", "lvis/lvis_v1_train+coco_fix.json"),
    "lvis_v1_train+coco_box": ("coco/", "lvis/lvis_v1_train+coco_box.json"),
    "lvis_v1_train+coco_taocats": ("coco/", "lvis/lvis_v1_train+coco_taocats.json"),
    "lvis_v1_train_person": ("coco/", "lvis/lvis_v1_train_person.json"),
    "lvis_v1_val_person": ("coco/", "lvis/lvis_v1_val_person.json"),
    "lvis_v1_train_person_id1": ("coco/", "lvis/lvis_v1_train_person_id1.json"),
    "lvis_v1_val_person_id1": ("coco/", "lvis/lvis_v1_val_person_id1.json"),
    "lvis_cocounlabeled_35.9_135": ("coco/unlabeled2017", "lvis/lvis_cocounlabeled_35.9_135.json"),
    "lvis_cocounlabeled_35.9_117": ("coco/unlabeled2017", "lvis/lvis_cocounlabeled_35.9_117.json"),
    "lvis_cocounlabeled_35.9_126": ("coco/unlabeled2017", "lvis/lvis_cocounlabeled_35.9_126.json"),
    "lvis_cocounlabeled_26.1_126": ("coco/unlabeled2017", "lvis/lvis_cocounlabeled_26.1_126.json"),
    "lvis_cocounlabeled_26.1_135": ("coco/unlabeled2017", "lvis/lvis_cocounlabeled_26.1_135.json"),
    "lvis_cocounlabeled_26.1_137": ("coco/unlabeled2017", "lvis/lvis_cocounlabeled_26.1_137.json"),
    "lvis_train_35.9_135": ("coco/", "lvis/lvis_train_35.9_135.json"),
    "lvis_train_35.9_126": ("coco/", "lvis/lvis_train_35.9_126.json"),
    "lvis_v1_train_halfcat_0": ("coco/", "lvis/lvis_v1_train_halfcat_0.json"),
    "lvis_v1_train_halfcat_1": ("coco/", "lvis/lvis_v1_train_halfcat_1.json"),
    "lvis_v1_val_halfcat_0": ("coco/", "lvis/lvis_v1_val_halfcat_0.json"),
    "lvis_v1_val_halfcat_1": ("coco/", "lvis/lvis_v1_val_halfcat_1.json"),
    "lvis_o365train_l35.9_135": ("objects365/train", "pseudo_labels/objects365_train_l35.9_135.json"),
    "lvis_o365val_l35.9_135": ("objects365/val", "pseudo_labels/objects365_val_l35.9_135.json"),
    "lvis_cocoun_l35.9_f1000": ("coco/unlabeled2017", "pseudo_labels/lvis_cocoun_l35.9_fill1000.json"), # thresh/images/anns: 125/99520/707509
    "lvis_o365train_l35.9_f1000": ("objects365/train", "pseudo_labels/objects365_train_l35.9_fill1000.json"), # thresh/images/anns: 125/99520/707509
    'imagenet_lvis': ('imagenet_lvis/', 'pseudo_labels/imagenet_lvis_image_info.json'),
    'imagenet_lvis_mini': ('imagenet_lvis_mini/', 'pseudo_labels/imagenet_lvis_image_info_mini.json'),
    'imagenet_lvis_100': ('imagenet_lvis/', 'pseudo_labels/imagenet_lvis_image_info_100.json'),
    "proposal_cocotrain_c50.6_5": ("coco/train2017", "pseudo_labels/proposals_cocotrain_c50.6_5.json"),
    "proposal_cocotrain_codist48.2l_5": ("coco/train2017", "pseudo_labels/proposals_cocotrain_codist48.2l_5.json"),
    'imagenet_lvis_l34.8_assl555': ('imagenet_lvis/', 'pseudo_labels/imagenet_lvis_l34.8_assl555.json'),
    'imagenet_lvis_l34.8_assl666': ('imagenet_lvis/', 'pseudo_labels/imagenet_lvis_l34.8_assl666.json'),
    'imagenet_lvis_l34.8_assl777': ('imagenet_lvis/', 'pseudo_labels/imagenet_lvis_l34.8_assl777.json'), # {'r': 632187, 'c': 384922, 'f': 494038}
    'imagenet_lvis_l34.8_777': ('imagenet_lvis/', 'pseudo_labels/imagenet_lvis_l34.8_777.json'),
    'imagenet_lvis_l34.8_888': ('imagenet_lvis/', 'pseudo_labels/imagenet_lvis_l34.8_777.json'), # 354k, 890k, freq_counts {'r': 105565, 'c': 157216, 'f': 627832}
}


for key, (image_root, json_file) in _PREDEFINED_SPLITS_LVIS_V1.items():
    register_lvis_v1_instances(
        key,
        get_lvis_v1_instances_meta(),
        os.path.join("datasets", json_file) if "://" not in json_file else json_file,
        os.path.join("datasets", image_root),
    )
