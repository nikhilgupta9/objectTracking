#Importing all the required packages
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
import datetime
from sunpy.io import jp2
import numpy as np
import glob, os
import sys
from PIL import Image
import datetime
from datetime import date, timedelta
from PIL import Image, ImageDraw
import astropy.units as u
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames

class objectTracking:

    def image_explorer(date_time_obj, base_url, image_wavelength, image_extension, threshold):
        """
        This function goes to the base_url directory and finds the closest image in time with the date_time_obj
        :param date_time_obj: timestamp of interest
        :param base_url: the url of the base directory
        :param image_wavelength: image wavelength as per the folder format
        :param image_extension: extension of the file
        :param threshold: threshold for closest image. Example: 120 seconds
        :return: returns the full path of that image
        """
        try:
            url = str(date_time_obj.year) + "/" + str(date_time_obj.month).zfill(2) + "/" + str(
                date_time_obj.day).zfill(2) + "/" + str(image_wavelength) + "/"
            os.chdir(base_url + url)
            imgs = glob.glob('*' + image_extension)
            times = [datetime.datetime.strptime(item.split('__SDO')[0], '%Y_%m_%d__%H_%M_%S_%f') for item in imgs]
            diff = [abs((date_time_obj - item).total_seconds()) for item in times]
            if (min(diff) < threshold):
                return base_url + url + imgs[diff.index(min(diff))]
            else:
                return 'No image found'
        except:
            return 'Error'

    def bbox_viz(img, X, Y):
        """
        This function takes image and solar event coordinate and then reutns the image object with a bounding-box
        :param img: image url of a fits file
        :param X: x coordinate arcsec
        :param Y: y coordinate arcsec
        :return: image object
        """
        h = jp2.get_header(img)
        header = h[0]
        image_w = header['NAXIS1']
        image_h = header['NAXIS2']
        center_x = header['CRPIX1']
        center_y = header['CRPIX2']
        scale_x = header['CDELT1']
        scale_y = header['CDELT2']
        PixX = (center_x + (X / scale_x))
        PixY = (center_y - (Y / scale_y))
        im = Image.open(img)
        offset = 100
        left = PixX - offset
        top = PixY - offset
        right = PixX + offset
        bottom = PixY + offset
        source_img = Image.open(img).convert("RGBA")
        draw = ImageDraw.Draw(source_img)
        draw.rectangle(((left, top), (right, bottom)), outline="yellow", width=4)
        return source_img

    def calculate_lon_delta(lat_degree, day):
        """
        This function takes two inputs lat_degree in degree and day in days. And then returns the shift in longitude
        at that latitude in the number of days given
        :param lat_degree: latitude in degree
        :param day: number of days
        :return: delta in  longitude degree
        """
        alpha = 14.11
        beta = -1.7
        gamma = -2.35
        velocity_in_deg = alpha + beta * (np.sin(np.deg2rad(lat_degree))) ** 2 + gamma * (
            np.sin(np.deg2rad(lat_degree))) ** 4
        delta_lon = velocity_in_deg * day
        return delta_lon

    def calculate_diff_in_days(t0, t1):
        """
        This function returns the number of days between two timestamps
        :param t0:
        :param t1:
        :return:
        """
        return (t0 - t1).total_seconds() / (24 * 60 * 60)


    def custom(timestamp, tbest, sig_Y):
        """
        This function returns the delta longitude at a given latitude
        :param timestamp: timestamp of intrest
        :param tbest: timestamp of the solar event
        :param sig_Y: y coordinate of the solar event
        :return: delta in longitude
        """
        d = calculate_diff_in_days(timestamp, tbest)
        new_X = round(calculate_lon_delta(sig_Y, d), 1)
        return new_X

    def conversion(sig_X, sig_Y, tbest, tnew):
        """
        This function takes four inputs. The coordinates of a solar event, time of the event and a new time of our interest.
        It takes these inputs in arcsecs and convert them into degree and then back to arcseconds
        :param sig_X: x coordinate of event in arcsec
        :param sig_Y: y coordinate of event in arcsec
        :param tbest: timestamp of the event
        :param tnew: timestamp of interest
        :return: new x coordinate at tnew in arcsec
        """
        tbest_date = str(tbest.date())
        tnew_date = str(tnew.date())
        del_X = custom(tnew, tbest, sig_Y)
        c = SkyCoord(sig_X * u.arcsec, sig_Y * u.arcsec, frame=frames.Helioprojective, obstime=tbest_date,
                     observer="earth")
        d = c.transform_to(frames.HeliographicStonyhurst)
        new_x_degree = (d.lon).value + del_X
        new_y_degree = (d.lat).value
        e = SkyCoord(new_x_degree * u.deg, new_y_degree * u.deg, frame=frames.HeliographicStonyhurst, obstime=tnew_date,
                     observer="earth")
        f = e.transform_to(frames.Helioprojective)
        new_X, new_Y = (f.Tx).value, (f.Ty).value
        return new_X

def main():
    # Strings
    base_url = '/data/AIA/'
    image_wavelength = 131
    image_extension = '.jp2'
    threshold = 120
    sig_TBEST = '2012-09-01T12:30:00.000'
    sig_START = '2012-08-30T21:00:00.000'
    sig_END = '2012-09-04T18:00:00.000'
    best = datetime.datetime.strptime(sig_TBEST, '%Y-%m-%dT%H:%M:%S.%f')
    start = datetime.datetime.strptime(sig_START, '%Y-%m-%dT%H:%M:%S.%f')
    end = datetime.datetime.strptime(sig_END, '%Y-%m-%dT%H:%M:%S.%f')
    sig_X = -29.9
    sig_Y = -98.3
    delta = 1 / 24

    #Creating a list of timestamps between start and end of the event with delta as cadence
    timestamps = []
    time = start
    while time < end:
        timestamps.append(time)
        time = time + datetime.timedelta(days=delta)

    #Creating a dataframe
    df = pd.DataFrame(timestamps, columns=['Timestamp'])
    df['Image'] = df.Timestamp.progress_apply(
        lambda x: image_explorer(x, base_url, image_wavelength, image_extension, threshold))
    df['Image_Timestamp'] = df.Image.progress_apply(
        lambda x: datetime.datetime.strptime(x.split('/')[-1].split('__SDO')[0], '%Y_%m_%d__%H_%M_%S_%f'))
    df['X'] = df.Image_Timestamp.progress_apply(lambda x: conversion(sig_Y, sig_X, best, x))

    #Saving all the images for a GIF
    for i in tqdm(range(len(df))):
        im = bbox_viz(df.iloc[i]['Image'], df.iloc[i]['X'], sig_Y)
        im = im.resize((1024, 1024), Image.ANTIALIAS)
        output_name = 'img_' + str(i) + '.png'
        im.save('/home/nguptagsu/objectTracking_images/' + output_name)

if __name__ == '__main__':
    main()
