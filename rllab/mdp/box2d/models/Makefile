all: hopper.xml cartpole.xml double_pendulum.xml half_cheetah.xml inverted_double_pendulum.xml car_parking.xml

hopper.xml: gen_xml.rb hopper.xml.rb
	ruby gen_xml.rb hopper.xml.rb > hopper.xml

half_cheetah.xml: gen_xml.rb half_cheetah.xml.rb
	ruby gen_xml.rb half_cheetah.xml.rb > half_cheetah.xml

cartpole.xml: gen_xml.rb cartpole.xml.rb
	ruby gen_xml.rb cartpole.xml.rb > cartpole.xml

double_pendulum.xml: gen_xml.rb double_pendulum.xml.rb
	ruby gen_xml.rb double_pendulum.xml.rb > double_pendulum.xml

inverted_double_pendulum.xml: gen_xml.rb inverted_double_pendulum.xml.rb
	ruby gen_xml.rb inverted_double_pendulum.xml.rb > inverted_double_pendulum.xml

car_parking.xml: gen_xml.rb car_parking.xml.rb
	ruby gen_xml.rb car_parking.xml.rb > car_parking.xml

test_cartpole:
	python cartpole.py

clean:
	rm *.xml
