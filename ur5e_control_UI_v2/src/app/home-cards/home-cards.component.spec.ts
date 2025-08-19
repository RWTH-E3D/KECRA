import { ComponentFixture, TestBed } from '@angular/core/testing';

import { HomeCardsComponent } from './home-cards.component';

describe('HomeCardsComponent', () => {
  let component: HomeCardsComponent;
  let fixture: ComponentFixture<HomeCardsComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ HomeCardsComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(HomeCardsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
