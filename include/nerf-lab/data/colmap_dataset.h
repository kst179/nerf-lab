#pragma once

#include <vector>
#include <string>
#include <map>

#include <boost/filesystem.hpp>
#include <opencv2/core.hpp>

#include "nerf-lab/data/dataset.h"
#include "nerf-lab/data/image_handle.h"

namespace nerf {
    
class ColmapDataset : public ImagesDataset {
public:
    ColmapDataset(fs::path root);
    // ~ColmapDataset();
    // ColmapDataset(const ColmapDataset& other) = delete;
    // ColmapDataset operator=(const ColmapDataset& other) = delete;

    cv::Mat load_image(int index) const;

    int image_name_to_id(const std::string& image_name) const { return image_name_to_id_.at(image_name); }
    std::string image_id_to_name(int image_id) const { return image_id_to_name_.at(image_id); }
    int image_to_camera_id(int image_id) const { return image_to_camera_id_.at(image_id); }
    
private:
    void try_load_db_();
    void try_load_images_();
    void try_load_cameras_();

    fs::path images_dir_;
    fs::path database_path_;
    
    fs::path cameras_file_;
    fs::path images_file_;
    fs::path points_txt_;

    bool database_found_;
    std::map<std::string, int> image_name_to_id_;
    std::map<int, std::string> image_id_to_name_;
    std::map<int, int> image_to_camera_id_;

    
};

}