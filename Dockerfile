# Start with the udacity tensorflow docker
# This already has tensorflow, jupyter and python running
FROM b.gcr.io/tensorflow-udacity/assignments:0.5.0

# Get some dependancies for python
RUN apt-get update \
	&& apt-get upgrade -y \
	&& apt-get install -y unzip wget build-essential \
		cmake git pkg-config libswscale-dev \
		python-dev \
		libtbb2 libtbb-dev libjpeg-dev \
		libpng-dev libtiff-dev libjasper-dev

# Upgrade pip
RUN pip install --upgrade pip

# Upgrade all pip components
RUN pip freeze --local | grep -v '^\-e' | cut -d = -f 1  | xargs -n1 pip install -U

# Install OpenCV
RUN pip install opencv-python