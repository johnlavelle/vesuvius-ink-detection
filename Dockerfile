FROM kaggle/python

# Install ngrok
RUN apt-get update && \
    apt-get install -y wget && \
    wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip && \
    unzip ngrok-stable-linux-amd64.zip && \
    rm ngrok-stable-linux-amd64.zip && \
    mv ngrok /usr/local/bin/ngrok && \
    chmod +x /usr/local/bin/ngrok

EXPOSE 4040
EXPOSE 80
EXPOSE 443

# Install other packages
RUN pip install shapely --upgrade
RUN pip install pytest rioxarray torcharrow torchsampler torchsummary zarr

# Start ngrok
#CMD ngrok http 80
