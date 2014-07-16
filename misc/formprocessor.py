#!/usr/bin/python

import os
import sys
import cgi
import cgitb; cgitb.enable()


def ucgiprint(inline='', unbuff=False, encoding='UTF-8'):
    """Print to the stdout.
    Includes keywords to define the output encoding (UTF-8 default, set to None to switch off encoding)
    and also whether we should flush the output buffer after every write (default not).
    """
    line_end = '\r\n'
    if encoding:
        inline = inline.encode(encoding)
        line_end = line_end.encode(encoding) # prob. not necessary as line endings will be the same in most encodings
    sys.stdout.write(inline)
    sys.stdout.write(line_end)
    if unbuff:
        sys.stdout.flush()           


def getform(theform, valuelist, notpresent='', nolist=False):
    """This function, given a CGI form as a FieldStorage instance, extracts the data from it, 
    based on valuelist passed in. Any non-present values are set to '' - although this can be changed.
    (e.g. to return None so you can test for missing keywords - where '' is a valid answer but to have the field missing isn't.)
    It also takes a keyword argument 'nolist'. If this is True list values only return their first value.
    """
    data = {}
    for field in valuelist:
        if not theform.has_key(field):                      #  if the field is not present (or was empty)
            data[field] = notpresent
        else:                                               # the field is present
            if  type(theform[field]) != type([]):           # is it a list or a single item
                data[field] = theform[field].value
            else:
                if not nolist:                               # do we want a list ?
                    data[field] = theform.getlist(field)     
                else:
                    data[field] = theform.getfirst(field)     # just fetch the first item
    return data


def isblank(indict):
    """Passed an indict of values it checks if any of the values are set.
    Returns True if the indict is empty, else returns False.
    I use it on the a form processed with getform
    to tell if my CGI has been activated without any form values.
    """
    for key in indict.keys():
        if indict[key]:
            return False
    return True


def replace(instring, indict):
    """A convenient way of doing multiple replaces in a single string.
    E.g. for html templates.
    Takes a string and a dictionary of replacements.
    In the dictionary - each key is replaced with it's value.
    We can also accept a list of tuples instead of a dictionary  (or anything accepted by the dict function)."""
    indict = dict(indict)
    for key in indict:
        instring = instring.replace(key, indict[key])
    return instring


# let's define out http/html values and templates
contentheader = 'Content-Type: text/html'

pagetemplate = '''
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml" lang="en" xml:lang="en">
    <head>
        <title>Form Processor Script</title>
        <meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />
		<style type="text/css">
			body {margin: 0 20px; padding: 0px}
		</style>
    </head>
    <body>
        <h1>Welcome to Form Processor, a Python CGI script</h1>
        **body**
		<br />
		<p><a href="http://www.python.org/"><img style="border:0" src="http://mgg1-2.local/~fpaolo/img/pythonpowered_small.gif" alt="Python Powered" /> </a></p>
    </body>
</html>
'''                     # note - the character encoding has been hardwired into the meta tag

htmlform = '''
<div class="form">
    <form action="**scriptname**" method="get">
        What is Your Name : <input name="name" type="text" value="**the name**" /><br />
        <input name="radio_param" type="radio" value="this"  **checked1** /> Select This<br />
        <input name="radio_param" type="radio"  value="that" **checked2** />or That<br />
        <input name="check1" type="checkbox" **checked3** /> Check This<br />
        <input name="check2" type="checkbox"  **checked4** /> and This Too ?<br />
        <input name="hidden_param" type="hidden" value="some_value" /><br />
        <input type="hidden" name="_charset_" />
        <input type="reset"  />
        <input type="submit" />
    </form>
</div>
'''

divider = '<hr />'

welcome = '''
<div class="welcome">
    <h2>Please Fill in Our Form</h2>
    <p>This CGI is an example CGI written for a <a href="http://www.pyzine.com">Pyzine</a> article.
</div>
'''

results = '''
<div class="results">
    <h2>You Submitted a Form</h2>
    <p>Your Name is "%s" (so you claim).</p>
    <p>You selected "%s" rather than "%s". (The radio buttons)</p>
    <p>You %s check "This". (first checkbox)</p>
    <p>You %s check "This Too". (second checkbox)</p>
    <p>A hidden value was sent - "%s".</p>
    <p>It was all sent in "%s" character encoding.</p>
</div>
'''

# let's set up some variables
scriptname = os.environ.get('SCRIPT_NAME', '')
checked = ' checked="checked" '

formvalues = ['name', 'radio_param', 'check1', 'check2', 'hidden_param', '_charset_']


def main():
    """This function forms the main body of the script."""
    ucgiprint(contentheader, encoding=None)                 # print the content header
    ucgiprint('', encoding=None)                            # followed by a blank line
    
    theform = cgi.FieldStorage()                            # use cgi to read and decode any form submission
    formdict = getform(theform, formvalues)                 # extract all our parameters from it
    
    if isblank(formdict):                                   # do we have a form submission to process ?
        displaywelcome()        # no form
    else:
        displayform(formdict)   # yes form !
        

def displaywelcome():
    """The script has been called without parameters.
    We should display the welcome message."""
    replacedict = {'**scriptname**' : scriptname,
                   '**the name**' : 'Fernando Paolo',
                   '**checked1**' : checked,
                   '**checked2**' : '',
                   '**checked3**' : checked,
                   '**checked4**' : checked
                   }
    
    thisform = replace(htmlform, replacedict)       # put the correct values into our form
    pagebody = welcome + thisform                   # these four lines could all be done in one step - but this is clearer
    wholepage = pagetemplate.replace('**body**', pagebody)
    ucgiprint(wholepage)


def displayform(formdict):
    """The script has been called with a form submission.
    Display the results of the form submission.
    """
    encoding = formdict['_charset_']  or 'UTF8'          # encoding may not be straightforward - let's find out what it is 
    thename = formdict['name'].decode(encoding)
    thisorthat = formdict['radio_param'].decode(encoding)
    this = formdict['check1'].decode(encoding)
    thistoo = formdict['check2'].decode(encoding)
    hidval = formdict['hidden_param'].decode(encoding)

    if thisorthat == 'this':
        check1 = checked
        check2 = ''
        radselect = 'This'
        unrad = 'That'
    else:
        check1 = ''
        check2 = checked
        radselect = 'That'
        unrad = 'This'
        
    if this == 'on':
        check3 = checked
        did1 = 'did'
    else:
        check3 = ''
        did1 = "didn't"

    if thistoo == 'on':
        check4 = checked
        did2 = 'did'
    else:
        check4 = ''
        did2 = "didn't"

    replacedict = {'**scriptname**' : scriptname,
                   '**the name**' : thename,
                   '**checked1**' : check1,
                   '**checked2**' : check2,
                   '**checked3**' : check3,
                   '**checked4**' : check4
                   }
    
    thisform = replace(htmlform, replacedict)       # put the previous values *back* into the form
    fullresults = results % (thename, radselect, unrad, did1, did2, hidval, encoding)
    pagebody = fullresults + divider + thisform                   
    wholepage = pagetemplate.replace('**body**', pagebody)
    ucgiprint(wholepage)


if __name__ == '__main__':
    main()
