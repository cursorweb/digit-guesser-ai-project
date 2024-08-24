from PIL import Image, ImageDraw, ImageOps

im_sm = Image.open("./number2.png")
im_sm.convert("L")

im_sm = ImageOps.invert(im_sm)


# ctx = ImageDraw.Draw(im_sm)
# ctx.rectangle((0, 0, 20 - 1, 20 - 1), outline=(123, 123, 123))

im = Image.new(mode="L", size=(28, 28))

im.paste(im_sm, box=(4, 4, 24, 24))

# im = im.transform(im.size, )
# print(im_sm.size)

im.save("number3.png")

# import matplotlib.pyplot as plt
# plt.imshow(im)
# plt.show()
