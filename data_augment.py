from PIL import Image, ImageEnhance


def scale(dir_path="examples/example1", name="1.jpg"):
    origin_path = dir_path + "/origin/" + name
    save_path = dir_path + "/scale/" + name
    pic = Image.open(origin_path)
    pic = pic.resize((int(3072 * 0.6), int(4096 * 0.6)))
    pic.save(save_path)


def rotate(dir_path="examples/example1", name="1.jpg"):
    origin_path = dir_path + "/origin/" + name
    save_path = dir_path + "/rotate/" + name
    pic = Image.open(origin_path)
    pic = pic.rotate(30)
    pic = pic.resize((int(3072 * 1.4), int(4096 * 1.4)))
    pic = pic.crop((600, 800, 600+3072, 800+4096))
    pic.save(save_path)


def bright(dir_path="examples/example1", name="1.jpg"):
    origin_path = dir_path + "/origin/" + name
    save_path = dir_path + "/bright/" + name
    pic = Image.open(origin_path)
    pic = ImageEnhance.Brightness(pic).enhance(1.5)
    pic.save(save_path)


if __name__ == '__main__':
    scale(dir_path="examples/example1", name="2.jpg")
    rotate(dir_path="examples/example1", name="2.jpg")
    bright(dir_path="examples/example1", name="2.jpg")

    scale(dir_path="examples/example2", name="5.jpg")
    rotate(dir_path="examples/example2", name="5.jpg")
    bright(dir_path="examples/example2", name="5.jpg")

    scale(dir_path="examples/example3", name="8.jpg")
    rotate(dir_path="examples/example3", name="8.jpg")
    bright(dir_path="examples/example3", name="8.jpg")

    scale(dir_path="examples/example4", name="11.jpg")
    rotate(dir_path="examples/example4", name="11.jpg")
    bright(dir_path="examples/example4", name="11.jpg")

    scale(dir_path="examples/example5", name="14.jpg")
    rotate(dir_path="examples/example5", name="14.jpg")
    bright(dir_path="examples/example5", name="14.jpg")
