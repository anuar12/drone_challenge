import requests

''' Simple script to download images from a file given by image-net.org '''

filename = 'farm.txt'
jpg_prepend = 'farm_'
num_lines = sum(1 for line in open('myfile.txt'))
print "Total number of image links in a file: ", num_lines
with open(filename, 'r') as f_urls:
    link = f_urls.readline()
    count = 0
    for i, link in enumerate(f_urls.readlines()):
        try:
            r = requests.get(link)
            if r.status_code == 200:
                count += 1
                name = "%05d.jpg" % count
                name = jpg_prepend + name
                with open(name, 'wb') as f:
                    f.write(requests.get(link).content)
                print "Downloaded image #", count
        except: pass
