import plotly.graph_objects as go
from plotly.subplots import make_subplots
from visdom import Visdom

## 
# ONLINE 
# Input:epochs_list:a list that includes epoch number; windows_id:the id of the window that will be used to show visualization in visdom, initialized_with None; 
# Training_losses_list:a list that includes loss values at each epoch; Dev_losses_list:a list that includes dev loss at each epoch
# Output: visualization at localhost:8097
# use visdom and plotly to visualize, requries these two packages and dependencies installed.
##
def visualize_with_visdom(epoch_list,training_losses,dev_losses,win_id=None):
    vis=Visdom()
    fig=make_subplots(rows=1,cols=2)
    fig.add_trace(
        go.Scatter(x=epoch_list, y=training_losses),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=epoch_list, y=dev_losses),
        row=1, col=2
    )
    fig.update_layout(height=600, width=800, title_text="Left:Training,Right:Validation")
    if(win_id!=None):
        vis.plotlyplot(fig,win=win_id)
        return None
    else:
        win_id=vis.plotlyplot(fig,win=win_id)
        return win_id
