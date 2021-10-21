from Applicaiton import Application
from Window import Window
from ImageManager import ImageManager


def main():
    app = Application()
    manager = ImageManager()
    window = Window("Image Re-colorization", 800, 600, manager)
    app.start()


if __name__ == "__main__":
    main()

