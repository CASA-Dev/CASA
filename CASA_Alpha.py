# -*- coding: utf-8 -*-
"""
Spyder Editor

@author: tom
"""
version = "CASA_alpha 0.3"

# ==============================================================================
# Iimported classes for program
# ==============================================================================
import pandas as pd
import numpy as np
from tkinter import *  # * does not import entire tkinter package
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import asksaveasfilename
import tkinter
from tkinter import Tk
import matplotlib
import os
import pdfkit
from jinja2 import Template
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # , NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
import datetime
import math
from functools import partial  # used for binding commands to buttons with paramater inputs

# implement the default mpl key bindings
# from matplotlib.backend_bases import key_press_handler

# ==============================================================================
# Color Scheme
# ==============================================================================
background_color = "black"
font = "Helvetica"
font_color = "green"
trough_color = "green"
border_color = "green"
#tet

# ==============================================================================
# Predefined global variables
# ==============================================================================
file_import = ""
price_factors_names = ['Garages', 'Bathrooms', 'Bedrooms', 'Acres', 'Square Feet',
                       'Fireplaces']  # these are the variables
#we care about. This will function as an "index" for our weighted values. ie.) weighted_factors[0] corresponds to the
#weighted value of the Garage is in position 0
price_factor_units = ['', '', '', 'acres', 'sqft', '']  # units for above factors

slider_weights_factors = np.zeros(price_factors_names.__len__())
weighted_total = np.sum(slider_weights_factors)
close_p = []
garage = []
full_bath = []
half_bath = []
bedrooms = []
acres = []
square_feet = []
fireplaces = []
year_built = []
age = []



# ==============================================================================
# Main class with _init_ function. when program is run:
# root = Tk()
# my_gui = CASAgui(root)
# root.mainloop()
#
# the CASAgui class is called, the rest of the work is done within the class and
# its functions.
# the _init_ function defines parameters for the window size, title, sets up
# the grid system so it is easy to put widgets in predictable locations, and
# adds button widget called "file" asking you to choose a file
# ==============================================================================
class CASAgui:
    class TestClass:
        def __init__(self):
            self.k = border_color

    def __init__(self):

        # Configure master root pane
        self.master = Tk()  # need to use self to use as an instance variable, otherwise scope of variable only local
        # to  __init__ functions
        self.master.title(version)
        self.master.geometry("800x1000+300+200")  # size + location
        self.master.config(background=background_color)
        self.master.grid_columnconfigure(0, weight=1)
        self.master.grid_columnconfigure(1, weight=1)  # make input boxes smaller
        self.master.grid_columnconfigure(2, weight=1)
        self.master.grid_columnconfigure(3, weight=1)
        self.master.grid_columnconfigure(4, weight=1)
        self.title_label = Label(self.master, text="CASA, THE FUTURE OF REAL ESTATE", font=(font, 20, "bold italic"),
                                 fg=font_color, background=background_color)
        self.title_label.grid(columnspan=5, row=0)
        self.label = Label(self.master, text="Please choose a CSV file to analyse", background=background_color,
                           fg=font_color)
        self.label.grid(columnspan=5, row=1)
        self.csv_button = Button(self.master, text="File", command=self.file_choose, background=background_color,
                                 highlightbackground=border_color, fg=font_color)
        self.csv_button.grid(columnspan=5, row=2)
        self.label = Label(self.master, text=file_import, fg=font_color, background=background_color)
        self.label.grid(columnspan=5, row=4)
        print('gridsize')
        print(self.master.grid_size())
        self.pie_chart = None
        self.input_entries = []
        self.estimated_price_label = None
        self.estimated_price = None
        self.master.mainloop()

    # ==============================================================================

    # Once "file" button is clicked, open file explorer to choose file, make sliding
    # scales visible and change name of file button to "change file"
    # ==============================================================================
    def file_choose(self):  # once file is uploaded it will show number sliders and name of doccument chosen
        global file_import
        file_import = askopenfilename(filetypes=(("CSV files", "*.csv"),))
        self.label.config(text=file_import)
        self.csv_button.config(text="Change File")

        tkinter.analyse = Button(text="Analyse", command=self.anaylse, background=background_color,
                                 highlightbackground=border_color, fg=font_color)
        tkinter.analyse.grid(columnspan=5, row=5)

        # scale0 = tkinter.Scale(label="Year Built", orient="horizontal", from_=0, to=10, command=self.print_value0, background=background_color, highlightbackground=border_color, fg=font_color, troughcolor=trough_color)
        # scale0.grid(column=0, row=6, sticky=E+W)


        # Adding label to clarify what the scales are used for
        self.labelScales = Label(self.master, text="Choose parameter weighting:", fg=font_color,
                                 background=background_color, font=("Helvetica", 17))
        self.labelScales.grid(column=2, row=6, columnspan=2)

        # ------Slider, Label, and Entry Discussion---------

        # alright lets rewrite this scale thing to be neat !
        # - use dictionary and unpack using ** to make configs identical
        # - store in list
        # -use lambda to simplfy "printvalue" function
        sliders= []
        slider_config={'orient':"horizontal", 'from_': 0, 'to': 10, 'background': background_color,
                'highlightbackground': border_color, 'fg': font_color,
                'troughcolor': trough_color}
        # get current # of rows
        grid_size = self.master.grid_size()
        curr_num_rows = grid_size[1]
        for i in range(0, price_factors_names.__len__()):
            # iterate through each slider
            curr_scale=tkinter.Scale(**slider_config)
            #command_wrapped=partial(self.print_value,curr_scale) # used to feed parameters to print command
            #  partials dont work here try lambda function ^
            curr_scale.config(label=price_factors_names[i],
                              command=lambda value, scale=curr_scale: self.print_value(value, scale))
            # wacky lambda function, essential lmabda function is just a temporary function without a name
            # used it above as a "go-between" to print_val, that way the assigned command (the lambda function)
            # is only taking one input which satisfies what tkinter.Scale() wants to spit into the command
            curr_scale.grid(column=2, row=curr_num_rows + i, sticky=E + W, columnspan=2)  # start at column 7

            # ----Also make  labels and input boxes for attributes of the property being appraised
            curr_label = Label(fg=font_color, background=background_color)
            label_text = "# of " + price_factors_names[i]
            curr_label.config(text=label_text)
            curr_label.grid(column=0, row=curr_num_rows + i)
            # ---set up input boxes------
            curr_entry = Entry(fg=font_color, background='gray11', width=12)
            curr_entry.grid(column=1, row=curr_num_rows + i, columnspan=1)
            self.input_entries.append(curr_entry)

    def generate_pichart(self):
        # ==============================================================================
        # When "analyse" button is clicked, takes data gathered from sliding scales
        # and shows user percent break down in an easy to interperet pie chart
        # =============================================================================

        # check if piechart has already been created

        if self.pie_chart:  # pie chart exists if its span has been defined
            # if the pie chart already exists, clear axis, redraw
            self.pie_chart.redraw()




        else:  # Create chart and button

            self.pie_chart = PieChart(self.master)
            self.pie_chart.draw()

            # add button

            grid_size = self.master.grid_size()
            curr_num_rows = grid_size[1]
            tkinter.report = Button(text="Generate Report", command=self.report, background=background_color,
                                    highlightbackground=border_color, fg=font_color)
            tkinter.report.grid(column=4, row=curr_num_rows)

            matplotlib.rcParams['text.color'] = 'black'

    def anaylse(self):
        # Calculates each factors price, generates a piechart , and calculates estimated appraisal price based on inputs

        self.generate_pichart()

        ################################testin#########################################
        weighted_total = sum(slider_weights_factors)
        csv_f = pd.read_csv(file_import)

        close_p_dirty = np.array(csv_f["Closed Price"])
        garage_dirty = np.array(csv_f["Garage"])
        full_bath_dirty = np.array(csv_f["# Full Baths"])
        half_bath_dirty = np.array(csv_f["# Half Baths"])
        bedrooms_dirty = np.array(csv_f["# of Bedrooms"])
        acres_dirty = np.array(csv_f["Acres"])
        square_feet_dirty = np.array(csv_f["Square_Feet"])
        fireplaces_dirty = np.array(csv_f["Num Fireplaces"])
        year_built_dirty = np.array(csv_f["Year Built"])
        # close_d_dirty = np.array(csv_f["Closed Date"]) # not in use right now, close date

        # creates array with locations of all "good" values, values that are not null or
        # stored as NaN or missing data

        good_elements = [i for i, x in enumerate(np.isnan(close_p_dirty)) if x == False]

        # removes all data associated with null values in close price


        global close_p  # reference global variables
        global garage
        global full_bath
        global half_bath
        global bedrooms
        global acres
        global square_feet
        global fireplaces
        global year_built
        global age
        # Removing "bad" data
        close_p = close_p_dirty[good_elements]
        garage = garage_dirty[good_elements]
        full_bath = full_bath_dirty[good_elements]
        half_bath = half_bath_dirty[good_elements]
        bedrooms = bedrooms_dirty[good_elements]
        acres = acres_dirty[good_elements]
        square_feet = square_feet_dirty[good_elements]
        fireplaces = fireplaces_dirty[good_elements]
        year_built = year_built_dirty[good_elements]
        #TODO This is super hardcoded , is there a way to fix this ? I think this may be the best we can do in the circumstance
        # close_d = close_d_dirty[good_elements] #not in use right now
        age = [datetime.date.today().year - x for x in year_built]
        bathrooms = ((half_bath * 0.5) + full_bath)

        close_p_average = np.average(
            close_p)  # this is not the WIEGHTED AVERAGE,just the overall average of close price

        # TODO Create coeffcient matrix

        good_data = [garage, bathrooms, bedrooms, acres, square_feet, fireplaces]  #this line is no bueno, hopefully we
        # Can make this more flexible

        # ------------Display Labels--------------
        # Display how much each attribute is worth, calculate the weighting of each
        #

        if sum(slider_weights_factors) != 0:
            # only calculate labels if they have been moved !
            weight_list = []  # make a nice way to store the weights caluclate
            dollar_labels = []
            for i in range(0, slider_weights_factors.__len__()):
                curr_label = Label(fg=font_color, background=background_color)
                unit = price_factor_units[i]
                curr_weight = calc_weights(good_data[i], slider_weights_factors[i], sum(slider_weights_factors),
                                           close_p_average)
                weight_list.append(curr_weight)

                if unit:
                    # if units exist print them
                    label_text = " = $" + '{0:,.2f}'.format(curr_weight, 2) + "/" + unit
                else:
                    # else dont print anything for units
                    label_text = " = $" + '{0:,.2f}'.format(curr_weight, 2)

                curr_label.config(text=label_text)
                curr_label.grid(column=4, row=7 + i)
                dollar_labels.append(curr_label)
            # display avg price
            close_p_label = Label(text=" = $" + '{0:,.2f}'.format(close_p_average, 2), fg=font_color,
                                  background=background_color)
            close_p_label.grid(column=4, row=13)

            weight_list = np.array(weight_list)

            # --------------Display estimated price
            #TODO this should probably be its own method
            appraisal_attributes = self.read_input_entries()
            if appraisal_attributes.size >= 1:
                # if all the inputs are filled in with number, estimate the price and show it
                estimated_price = estimate_price(appraisal_attributes, weight_list)

                grid_size = self.master.grid_size()
                curr_num_rows = grid_size[1]
                self.estimated_price_label = Label(text="Estimated Price = $" + '{0:,.2f}'.format(estimated_price, 2),
                                                   fg=font_color,
                                                   background=background_color)
                self.estimated_price_label.grid(column=0, row=curr_num_rows - 1)
                # TODO add it so when the user enters a new input the estimated value changes
                self.estimated_price = estimated_price
                print(estimated_price)
            else:
                # tell user that they are a mess and they need to get their shit together....summer
                print('get your shit together, put in numbersS')

    def read_input_entries(self):
        # converts entries (user input) to doubles, then outputs values in a np.array
        # this is used to "read" the current data entered and ultimately estimate price
        # outputs empty np.array
        entry_num = []
        flag = 0  # to make sure all the inputs are convertible numbers
        for entry in self.input_entries:
            curr_string = entry.get()
            # get rid of commas
            curr_string = curr_string.replace(',', '')

            if is_number(curr_string):
                # convert to double'
                entry_num.append(float(curr_string))
            else:
                flag = 1

        if flag == 1:
            # return empty array if not all are numbers
            return np.array([])
        else:
            # convert to np array
            return np.array(entry_num)

    def print_value(self, val, slider):
        # ==============================================================================
        # takes values from sliding scales and stores them globally

        # used anywhere they are needed in the program
        # ==============================================================================

        param_name = slider.cget('label') # get name of varibale tbe attached slider is changing
        param_index = price_factors_names.index(param_name) #find what index corresponds to this particular var
        slider_weights_factors[param_index] = int(val)
        self.anaylse()





    def report(self):
        # ==============================================================================

        #  Report will create all charts and graphs and organize them in a pdf file
        # ==============================================================================
        # TODO Most of this is hardcoded in, anyway to make more flexible?
        # Plot figure with subplots of different sizes
        fig = plt.figure(1)
        # set up subplot grid
        gridspec.GridSpec(3, 4)
        print("report")
        # create bins for histogram based on data entered
        num_of_bins = 10
        price_step = max(close_p) / num_of_bins
        # steps should be numbers that are "nicer" to look at ie , every 10k, 25k, 50k, or 100k
        # we'll use steps that are divisible by 10,000 ie.) 20k,30, 40k etc that won't limit our data domain
        price_step = math.ceil(price_step / 10000) * 10000
        price_bins = []
        price_bins_label = []
        i = 0
        j = 0
        while i < num_of_bins + 1:  # write that data to two arrays
            price_bins.append(price_step * i)
            price_bins_label.append(price_bins[j] / 1000)
            i = i + 1
            j = j + 1

        # suplot for CPD
        CPDplt = plt.subplot2grid((3, 4), (0, 0), colspan=2, rowspan=3)
        # plt.locator_params(axis='x', nbins=11)
        # plt.locator_params(axis='y', nbins=5)
        plt.title('Close Price Distribution')
        plt.xlabel('Close Price (thousands)')
        plt.ylabel('Properties')
        calculatedAppraisalPrice = self.estimated_price
        plt.hist([close_p, calculatedAppraisalPrice], bins=price_bins, stacked=True, color=["gray", "red"],
                 label={'', "Calcuated Appraisal Value"})
        plt.xticks(price_bins, price_bins_label)
        plt.legend()
        # plt.savefig('plot1.jpg') #  remove or supress this


        # ----subplot for ACRES
        #fig= plt.figure(2)
        # again going to "prettify" this data to be resolved to either multiples of .1 or .25 of an acre
        num_of_acre_bins = 10
        acre_step_divide = max(acres) / num_of_acre_bins
        if acre_step_divide < .1:
            # step by .1
            acre_step = .1
        elif acre_step_divide > .1 and acre_step_divide < .25:
            # step by .25
            acre_step = .25
        elif acre_step_divide > .25 and acre_step_divide < .5:
            # step by .5
            acre_step = .5
        else:
            acre_step = math.ceil(acre_step_divide)  # going by steps that are whole numbers

        acre_bins = []
        i = 0
        # write that data to two arrays
        while i < 11:
            acre_bins.append(acre_step * i)
            i = i + 1

        #print(acre_bins)
        ## small subplot 1
        plt.subplot2grid((3, 4), (0, 2), colspan=2)
        # plt.locator_params(axis='x', nbins=5)
        # plt.locator_params(axis='y', nbins=5)
        plt.title('Acre Distribution')
        plt.xlabel('Acres')
        plt.ylabel('Properties')
        plt.hist(acres, bins=acre_bins, color='b')
        plt.xticks(acre_bins)
        #plt.savefig('plot2.jpg') # remove this/ supress

        ## Setup for GLA Distribution
        square_feet_step = max(square_feet) / 10.0
        square_feet_bins = []
        i = 1
        # write that data to two arrays
        while i < 11:
            square_feet_bins.append(square_feet_step * i)
            i = i + 1

        # GLA PLOT--------------------
        #fig = plt.figure(3)
        plt.subplot2grid((3, 4), (1, 2), colspan=2)
        # plt.locator_params(axis='x', nbins=5)
        # plt.locator_params(axis='y', nbins=5)
        plt.title('GLA Distribution')
        plt.xlabel('GLA')
        plt.ylabel('Properties')
        plt.hist(square_feet, bins=square_feet_bins, color='r')
        plt.xticks(square_feet_bins)
        #plt.savefig('plot3.jpg')  #  remove or supress this



        ## Setup data for  AGE DISTRB plot
        age_step = max(age) / 10.0
        age_bins = []
        i = 1
        # write that data to two arrays
        while i < 11:
            age_bins.append(age_step * i)
            i = i + 1

        print(acre_bins)
        ## small subplot 1



        ## AgeDistrub plot
        #fig = plt.figure(4)
        plt.subplot2grid((3, 4), (2, 2), colspan=2)
        # plt.locator_params(axis='x', nbins=5)
        # plt.locator_params(axis='y', nbins=5)
        plt.title('Age Distribution')
        plt.xlabel('Age')
        plt.ylabel('Properties')
        plt.hist(age, bins=age_bins, color='g')
        plt.xticks(age_bins)
        #plt.savefig('plot4.jpg')  # TODO remove or supress this


        ## Show figure
        fig.tight_layout()
        fig.set_size_inches(w=11, h=8.5)  # this portable? Need to make window
        # big so the text converts to pdf on a good scale
        # These numbers were arrived at by trial and error, definitely not best solution
        # TODO make this portable/ not hardcoded

        # Make full screen so text prints to pdf in right format
        # mng = plt.get_current_fig_manager()
        # mng.window.state('zoomed')

        #Slap inside TK so we can add a "make pdf button
        figRoot = tkinter.Tk()
        figRoot.wm_title("Figures for X")  # TODO fix X
        canvas = FigureCanvasTkAgg(fig, master=figRoot)
        canvas._tkcanvas.pack(expand=1)
        canvas.show()

        # add button :D

        pdfButton = Button(master=figRoot, text="Create PDF")
        buttonActionWrapped = partial(save2pdf, fig, pdfButton)  # this line is so we can add params to button callback
        pdfButton.config(
            command=buttonActionWrapped)  # adds command, in this specific order so the button is itself a parameter
        pdfButton.pack()
        # fig.savefig("Test.pdf")
        # resize
        # to get correct looking text , set to full screen



def save2pdf(figure, button):
    # this method is used for saving figures to 1 page pdf form.
    # inputs are the root of all the figures and the button used for saving
    #
    # This function accomplishes by the following steps
    # 1.) Save figure as a png
    # 2.) Populates html template with image and data
    # 3.) converts html to pdf


    pdf_name = asksaveasfilename(filetypes=[('PDF', '*.pdf')])
    # check if error or cancel button, if canceled returns empty string.Empty strings are falsy
    if pdf_name:
        # 'if not empty'
        # 1.) Save fig as png
        figure.savefig('pdf_build_resources/plots/plot.png')

        # 2.) Populate html template
        fill_html_template()

        #3 convert html to pdf and save to pdf_name TODO
        pdfkit.from_file('pdf_build_resources/rendered.html', pdf_name)

        button.config(text='Sucess!')  # just to stop people (aileen from saving like 100 copies
        # if there wasn't an error the next line should execute (I believe) therefore if it
        # executes it means it was a success


# ==============================================================================
# ==============================================================================

# ==============================================================================
#                                   debugging
# ==============================================================================
# ==============================================================================
# print("good elements = " + str(good_elements))
# print("\nClose price Pandas = \n" + str(csv_f["Closed Price"]))
# print("\nclose_price_dirty Numpy array = " + str(close_p_dirty))
# print("\nisnan showing NaN on close_p_dirty = " + str(np.isnan(close_p_dirty)))
# print("\nclose_p = " + str(close_p))
# print("\n Length of good_elements = " + str(len(good_elements)))
# print("\n Length of close_p = " + str(len(close_p)))
# print("\n Length of garage = " + str(len(garage)))
# print("\n Length of full_bath = " + str(len(full_bath)))
# print("\n Length of half_bath = " + str(len(half_bath)))
# print("\n Length of bedrooms = " + str(len(bedrooms)))
# print("\n Length of acres = " + str(len(acres)))
# print("\n Length of square_feet = " + str(len(square_feet)))
# #print("\n Length of close_d = " + str(len(close_d)))
# print(close_d)
# print(close_d_dirty)
# ==============================================================================
# #############################################################################
# ==============================================================================


def fill_html_template():
    # function is used to populate the html template used to generate pdfs
    # ----open and read template'

    file_template = open('pdf_build_resources/template.html',
                         "r")  # opens file with name of "test.txt" # TODO will this work on windows?
    html_template = Template(file_template.read())
    file_template.close()

    # ------ load figures names
    plot_list = os.listdir('pdf_build_resources/plots')  # this was used for when inserting multiple pictures

    # -------- map variables
    context = {'plot_list': plot_list}

    # ------write to file
    file_rendered = open('pdf_build_resources/rendered.html', 'w')
    file_rendered.write(html_template.render(context))
    file_rendered.close()


def calc_r_squared(actual_price_comps, estimate_price_comps):
    # this calculates the r^2 value between the ideal case (actual price ==estimated price) which is a line of slope
    # 1 and of the scatter data of (actual price, our estimated price) for all the comps
    # for simplicity make sure the inputs are np column vectors
    # SS Error (total "area" from your best fit line sum((y-y_hat)^2)
    y = estimate_price_comps
    y_hat = actual_price_comps  # because of slope 1
    distance = y - y_hat
    area = distance ** 2  # elementwise squaring
    ss_error = sum(area)

    # ss_total, total "area" from avg to scatter data
    avg = np.mean(estimate_price_comps)
    # create column vector of avg (or just write a for loop)
    ones = np.ones((y.size(), 1))
    y_bar = avg * ones
    area = (y - y_bar) ** 2
    ss_total = sum(area)

    return 1 - ss_error / ss_total  # formula for r^2 # TODO haven't tested this yet


def numerical_jacobian(function, x):
    # This function calculates the Jacobian of function using finite differences
    # kinda cool but it looks like you can input functions as arguments in python, kinda trippy

    delta_x = min(x) * .1 + .1 * int(min(x) == 0)  # defining delta x so it can never be 0
    f_0 = function(x)

    cols = x.__len__()
    rows = f_0.__len__()
    jacobian = np.zeros((rows, cols))
    for i in range(0, cols):
        delta_x = x[i] * .5 + .05 * int(x[i] == 0)  # define delta_x relatively, make sure its not equal to zero
        x_n = x
        x_n[i] = x[i] + delta_x  # perturb
        f_n = function(x_n)
        jacobian[:, i] = (f_n - f_0) / (delta_x)  # replace the column
        # TODO test this bad boy
    return jacobian


def my_fun(x, compdata):
    # This function is the derivative of how ¨close¨ our data is to our best fit hyper plane, finding the root this will
    # help us find the extrema , and ultimately help us create a best fit plane ie.) linear regression analysis
    f_x = None  # TODO finish this puppy


def estimate_price(property_attributes, weights):
    # This function is used to calculate the estimate price based on the weights
    # super simple function, only included for resuseability / readability purposes

    return np.dot(property_attributes, weights)


def calc_weights(comp_attribute_list, slider_weight, total_slider_weight, close_p_average):
    # calculating values based on average price and average number of each attribute
    # ie.) cost per acre = avg_price* %importance_of_acres / (number of acres)
    # This function effectively calculates the coefficients for the hyper plane
    avg_num_attribute = np.average(comp_attribute_list)  # average of the attribute
    weighted_price = (close_p_average / avg_num_attribute) * (slider_weight / total_slider_weight)
    print(avg_num_attribute)

    return weighted_price


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

class PieChart:
    def __init__(self, master):
        self.row = None
        self.figure = None
        self.ax = None
        self.canvas = None
        self.values = slider_weights_factors
        self.column = 0  # start on left most column and span page
        self.span, garbage = master.grid_size()  # num of columns to span
        self.master = master
        self.labels = price_factors_names

    def draw(self):
        # if a piechart doesn't exist create a new one
        matplotlib.rcParams[
            'text.color'] = font_color  # changes global defaults for text colors for matplotlib,
        # only in this function. only way i knnow to change pie chart label colors #TODO

        f = Figure(figsize=(5, 5), facecolor=background_color)
        self.figure = f

        a = f.add_subplot(111)
        self.ax = a

        f.text(0.5, 0.97, "Attributes Weighted", style='italic', horizontalalignment='center',
               verticalalignment='top',
               fontdict=None, fontsize=22, color=font_color)  # font_color is module scoped variable
        labels = self.labels.copy()
        explode = list()
        for k in range(0, price_factors_names.__len__()):
            explode.append(0.1)
            # don't show names of variables not weighted
            if slider_weights_factors[k] < .01:
                # == 0 but factoring in for some rounding
                labels[k] = ''

        patches, texts, autotexts = self.ax.pie(self.values, labels=labels, explode=explode, shadow=True,
                                                autopct='%1.1f%%',
                                                colors=("r", "b", "c", "m", "y", "g", "w", "0.75"))
        for t in autotexts:  # sets autopct color to white (percentages inside of wedges)
            # http://matplotlib.org/examples/pylab_examples/pie_demo2.html
            t.set_color('black')

        self.canvas = FigureCanvasTkAgg(f, master=self.master)

        self.figure = f  # store for access outside this method
        cols, master_rows = self.master.grid_size()
        self.row = master_rows + 2

        self.ax = a
        self.canvas.get_tk_widget().grid(column=0,
                                         row=self.row, columnspan=self.span,
                                         rowspan=self.span, sticky=N + E + S + W)
        # start 2 rows after sliders and labels end

        self.canvas.show()
        matplotlib.rcParams['text.color'] = 'black'

    def redraw(self):
        self.values = slider_weights_factors  # updates values, not necessary due to how python points to lists but included
        # for clarity
        self.ax.clear()
        matplotlib.rcParams[
            'text.color'] = font_color
        labels = self.labels.copy()
        explode = list()
        for k in range(0, price_factors_names.__len__()):
            explode.append(0.1)
            # don't show names of variables not weighted
            if slider_weights_factors[k] < .01:
                # == 0 but factoring in for some rounding
                labels[k] = ''

        patches, texts, autotexts = self.ax.pie(self.values, labels=labels, explode=explode, shadow=True,
                                                autopct='%1.1f%%',
                                                colors=("r", "b", "c", "m", "y", "g", "w", "0.75"))
        for t in autotexts:  # sets autopct color to white (percentages inside of wedges) http://matplotlib.org/examples/pylab_examples/pie_demo2.html
            t.set_color('black')

        self.canvas.draw()
        matplotlib.rcParams['text.color'] = 'black'

    def __bool__(self):

        return bool(self.canvas)  # returns true if canvas exists




my_gui = CASAgui()
