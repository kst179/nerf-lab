#pragma once

#include <vector>

#include <boost/filesystem.hpp>
#include <opencv2/core.hpp>

#include "nerf-lab/data/image_handle.h"
#include "nerf-lab/data/camera_model.h"

namespace nerf{

namespace fs = boost::filesystem;

class ImagesDataset {
public:
    virtual ~ImagesDataset() {};
    ImagesDataset() {};

    fs::path root() const { return root_; };
    int size() const { return images_paths_.size(); }

    virtual cv::Mat load_image(int image_id) const = 0;

    std::vector<std::string> image_names() const {
        std::vector<std::string> image_names;
        image_names.reserve(images_paths_.size());

        for (const fs::path& path : images_paths_) {
            image_names.push_back(path.filename().string());
        }

        return image_names;
    };

    std::string get_image_name(int index) const {
        return images_paths_[index].filename().string();
    };

    bool has_images() const { return !images_.empty(); }
    bool has_cameras() const { return !images_.empty(); }

    const std::map<int, std::shared_ptr<ImageHandle>>& images() const { return images_; }
    const std::map<int, std::shared_ptr<Camera>>& cameras() const { return cameras_; }

    std::shared_ptr<ImageHandle> get_image(int id) const { return images_.at(id); }
    std::shared_ptr<Camera> get_camera(int id) const { return cameras_.at(id); }

    std::vector<int> image_ids() const {
        std::vector<int> ids;
        ids.reserve(images_.size());
        for (auto& [id, image] : images_) {
            ids.push_back(id);
        }

        return ids;
    }

protected:
    fs::path root_;
    std::vector<fs::path> images_paths_;

    std::map<int, std::shared_ptr<ImageHandle>> images_;
    std::map<int, std::shared_ptr<Camera>> cameras_;
};

}