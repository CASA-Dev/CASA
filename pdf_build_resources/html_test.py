

from jinja2 import Template
import os

def fill_html_template():

    # ----open and read template
    file_template = open("template.html", "r")  # opens file with name of "test.txt" # TODO will this work on windows?
    html_template = Template(file_template.read())
    file_template.close()

    #------ load figures names
    plot_list = os.listdir('plots')

    #-------- map variables
    context = {'plot_list': plot_list}


    #------write to file
    file_rendered = open('rendered_html', 'w')
    file_rendered.write(html_template.render(context))
    file_rendered.close()



